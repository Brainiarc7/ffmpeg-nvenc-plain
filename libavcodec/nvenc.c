/*
 * H.264 hardware encoding using nvidia nvenc
 * Copyright (c) 2014 Timo Rothenpieler <timo@rothenpieler.org>
 *
 * This file is part of FFmpeg.
 *
 * FFmpeg is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * FFmpeg is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with FFmpeg; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
 */

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#include <nvEncodeAPI.h>

#include "libavutil/internal.h"
#include "libavutil/imgutils.h"
#include "libavutil/avassert.h"
#include "libavutil/opt.h"
#include "libavutil/mem.h"
#include "avcodec.h"
#include "internal.h"

#ifdef _WIN32
#define CUDAAPI __stdcall
#else
#define CUDAAPI
#endif

typedef enum cudaError_enum {
    CUDA_SUCCESS = 0
} CUresult;
typedef int CUdevice;
typedef void* CUcontext;

typedef CUresult(CUDAAPI *PCUINIT)(unsigned int Flags);
typedef CUresult(CUDAAPI *PCUDEVICEGETCOUNT)(int *count);
typedef CUresult(CUDAAPI *PCUDEVICEGET)(CUdevice *device, int ordinal);
typedef CUresult(CUDAAPI *PCUDEVICEGETNAME)(char *name, int len, CUdevice dev);
typedef CUresult(CUDAAPI *PCUDEVICECOMPUTECAPABILITY)(int *major, int *minor, CUdevice dev);
typedef CUresult(CUDAAPI *PCUCTXCREATE)(CUcontext *pctx, unsigned int flags, CUdevice dev);
typedef CUresult(CUDAAPI *PCUCTXPOPCURRENT)(CUcontext *pctx);
typedef CUresult(CUDAAPI *PCUCTXDESTROY)(CUcontext ctx);

typedef NVENCSTATUS (NVENCAPI* PNVENCODEAPICREATEINSTANCE)(NV_ENCODE_API_FUNCTION_LIST *functionList);

static const GUID dummy_license = { 0x0, 0x0, 0x0, { 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0 } };

static PCUINIT cu_init = 0;
static PCUDEVICEGETCOUNT cu_device_get_count = 0;
static PCUDEVICEGET cu_device_get = 0;
static PCUDEVICEGETNAME cu_device_get_name = 0;
static PCUDEVICECOMPUTECAPABILITY cu_device_compute_capability = 0;
static PCUCTXCREATE cu_ctx_create = 0;
static PCUCTXPOPCURRENT cu_ctx_pop_current = 0;
static PCUCTXDESTROY cu_ctx_destroy = 0;

static int nvenc_init_count;
static NV_ENCODE_API_FUNCTION_LIST nvenc_funcs;
static NV_ENCODE_API_FUNCTION_LIST *p_nvenc = 0;
static int nvenc_device_count = 0;
static CUdevice nvenc_devices[16];
static unsigned int nvenc_use_device_id = 0;

#ifdef _WIN32
#define LOAD_FUNC(l, s) GetProcAddress(l, s)
#define DL_CLOSE_FUNC(l) FreeLibrary(l)
static HMODULE cuda_lib;
static HMODULE nvenc_lib;
#else
#define LOAD_FUNC(l, s) dlsym(l, s)
#define DL_CLOSE_FUNC(l) dlclose(l)
static void *cuda_lib;
static void *nvenc_lib;
#endif

#define ifav_log(...) if (avctx) { av_log(__VA_ARGS__); }

typedef struct NvencInputSurface
{
    NV_ENC_INPUT_PTR input_surface;
    int width;
    int height;

    int lockCount;

    NV_ENC_BUFFER_FORMAT format;
} NvencInputSurface;

typedef struct NvencOutputSurface
{
    NV_ENC_OUTPUT_PTR output_surface;
    int size;

    NvencInputSurface *input_surface;

    int busy;
} NvencOutputSurface;

typedef struct NvencOutputSurfaceList
{
    NvencOutputSurface *surface;
    struct NvencOutputSurfaceList *next;
} NvencOutputSurfaceList;

typedef struct NvencTimestampList
{
    int64_t timestamp;
    struct NvencTimestampList *next;
} NvencTimestampList;

typedef struct NvencContext
{
    AVClass *avclass;

    NV_ENC_INITIALIZE_PARAMS init_encode_params;
    NV_ENC_CONFIG encode_config;
    CUcontext cu_context;

    int max_surface_count;
    NvencInputSurface *input_surfaces;
    NvencOutputSurface *output_surfaces;

    NvencOutputSurfaceList *output_surface_queue;
    NvencOutputSurfaceList *output_surface_ready_queue;
    NvencTimestampList *timestamp_list;
    int64_t last_dts;

    void *nvencoder;

    char *preset;
    int cbr;
    int twopass;
    int gobpattern;
} NvencContext;

#define CHECK_LOAD_FUNC(t, f, s) \
do { \
    f = (t)LOAD_FUNC(cuda_lib, s); \
    if (!f) { \
        ifav_log(avctx, AV_LOG_FATAL, "Failed loading %s from CUDA library\n", s); \
        goto error; \
    } \
} while(0)

static int nvenc_dyload_cuda(AVCodecContext *avctx)
{
    if (cuda_lib)
        return 1;

#if defined(_WIN32)
    cuda_lib = LoadLibrary(TEXT("nvcuda.dll"));
#elif defined(__CYGWIN__)
    cuda_lib = dlopen("nvcuda.dll", RTLD_LAZY);
#else
    cuda_lib = dlopen("libcuda.so", RTLD_LAZY);
#endif

    if (!cuda_lib) {
        ifav_log(avctx, AV_LOG_FATAL, "Failed loading CUDA library\n");
        goto error;
    }

    CHECK_LOAD_FUNC(PCUINIT, cu_init, "cuInit");
    CHECK_LOAD_FUNC(PCUDEVICEGETCOUNT, cu_device_get_count, "cuDeviceGetCount");
    CHECK_LOAD_FUNC(PCUDEVICEGET, cu_device_get, "cuDeviceGet");
    CHECK_LOAD_FUNC(PCUDEVICEGETNAME, cu_device_get_name, "cuDeviceGetName");
    CHECK_LOAD_FUNC(PCUDEVICECOMPUTECAPABILITY, cu_device_compute_capability, "cuDeviceComputeCapability");
    CHECK_LOAD_FUNC(PCUCTXCREATE, cu_ctx_create, "cuCtxCreate_v2");
    CHECK_LOAD_FUNC(PCUCTXPOPCURRENT, cu_ctx_pop_current, "cuCtxPopCurrent_v2");
    CHECK_LOAD_FUNC(PCUCTXDESTROY, cu_ctx_destroy, "cuCtxDestroy_v2");

    return 1;

error:

    if (cuda_lib)
        DL_CLOSE_FUNC(cuda_lib);

    cuda_lib = NULL;

    return 0;
}

static int check_cuda_errors(AVCodecContext *avctx, CUresult err, const char *func)
{
    if (err != CUDA_SUCCESS) {
        ifav_log(avctx, AV_LOG_FATAL, ">> %s - failed with error code 0x%x\n", func, err);
        return 0;
    }
    return 1;
}
#define check_cuda_errors(f) if (!check_cuda_errors(avctx, f, #f)) goto error

static int nvenc_check_cuda(AVCodecContext *avctx)
{
    int deviceCount = 0;
    CUdevice cuDevice = 0;
    char gpu_name[128];
    int smminor = 0, smmajor = 0;
    int i, smver;

    if (!nvenc_dyload_cuda(avctx))
        return 0;

    if (nvenc_device_count > 0)
        return 1;

    check_cuda_errors(cu_init(0));

    check_cuda_errors(cu_device_get_count(&deviceCount));

    if (!deviceCount) {
        ifav_log(avctx, AV_LOG_FATAL, "No CUDA capable devices found\n");
        goto error;
    }

    ifav_log(avctx, AV_LOG_VERBOSE, "%d CUDA capable devices found\n", deviceCount);

    nvenc_device_count = 0;

    for (i = 0; i < deviceCount; ++i) {
        check_cuda_errors(cu_device_get(&cuDevice, i));
        check_cuda_errors(cu_device_get_name(gpu_name, sizeof(gpu_name), cuDevice));
        check_cuda_errors(cu_device_compute_capability(&smmajor, &smminor, cuDevice));

        smver = (smmajor << 4) | smminor;

        ifav_log(avctx, AV_LOG_VERBOSE, "[ GPU #%d - < %s > has Compute SM %d.%d, NVENC %s ]\n", i, gpu_name, smmajor, smminor, (smver >= 0x30) ? "Available" : "Not Available");

        if (smver >= 0x30)
            nvenc_devices[nvenc_device_count++] = cuDevice;
    }

    if (!nvenc_device_count) {
        ifav_log(avctx, AV_LOG_FATAL, "No NVENC capable devices found\n");
        goto error;
    }

    return 1;

error:

    nvenc_device_count = 0;

    return 0;
}

static int nvenc_dyload_nvenc(AVCodecContext *avctx)
{
    PNVENCODEAPICREATEINSTANCE nvEncodeAPICreateInstance = 0;
    NVENCSTATUS nvstatus;

    if (!nvenc_check_cuda(avctx))
        return 0;

    if (p_nvenc) {
        nvenc_init_count++;
        return 1;
    }

#if defined(_WIN32)
    if (sizeof(void*) == 8) {
        nvenc_lib = LoadLibrary(TEXT("nvEncodeAPI64.dll"));
    } else {
        nvenc_lib = LoadLibrary(TEXT("nvEncodeAPI.dll"));
    }
#elif defined(__CYGWIN__)
    if (sizeof(void*) == 8) {
        nvenc_lib = dlopen("nvEncodeAPI64.dll", RTLD_LAZY);
    } else {
        nvenc_lib = dlopen("nvEncodeAPI.dll", RTLD_LAZY);
    }
#else
    nvenc_lib = dlopen("libnvidia-encode.so.1", RTLD_LAZY);
#endif

    if (!nvenc_lib) {
        ifav_log(avctx, AV_LOG_FATAL, "Failed loading the nvenc library\n");
        goto error;
    }

    nvEncodeAPICreateInstance = (PNVENCODEAPICREATEINSTANCE)LOAD_FUNC(nvenc_lib, "NvEncodeAPICreateInstance");

    if (!nvEncodeAPICreateInstance) {
        ifav_log(avctx, AV_LOG_FATAL, "Failed to load nvenc entrypoint\n");
        goto error;
    }

    p_nvenc = &nvenc_funcs;
    memset(p_nvenc, 0, sizeof(NV_ENCODE_API_FUNCTION_LIST));
    p_nvenc->version = NV_ENCODE_API_FUNCTION_LIST_VER;

    nvstatus = nvEncodeAPICreateInstance(p_nvenc);

    if (nvstatus != NV_ENC_SUCCESS) {
        ifav_log(avctx, AV_LOG_FATAL, "Failed to create nvenc instance\n");
        goto error;
    }

    ifav_log(avctx, AV_LOG_VERBOSE, "Nvenc initialized successfully\n");

    nvenc_init_count = 1;

    return 1;

error:
    if (nvenc_lib)
        DL_CLOSE_FUNC(nvenc_lib);

    nvenc_lib = NULL;
    p_nvenc = NULL;
    nvenc_init_count = 0;

    return 0;
}

static void nvenc_unload_nvenc(AVCodecContext *avctx)
{
    if (nvenc_init_count <= 0)
        return;

    nvenc_init_count--;

    if (nvenc_init_count > 0)
        return;

    DL_CLOSE_FUNC(nvenc_lib);
    nvenc_lib = NULL;
    p_nvenc = NULL;

    nvenc_device_count = 0;

    DL_CLOSE_FUNC(cuda_lib);
    cuda_lib = NULL;

    cu_init = NULL;
    cu_device_get_count = NULL;
    cu_device_get = NULL;
    cu_device_get_name = NULL;
    cu_device_compute_capability = NULL;
    cu_ctx_create = NULL;
    cu_ctx_pop_current = NULL;
    cu_ctx_destroy = NULL;

    ifav_log(avctx, AV_LOG_VERBOSE, "Nvenc unloaded\n");
}

static void out_surf_queue_enqueue(NvencOutputSurfaceList** head, NvencOutputSurface *surface)
{
    if (!*head) {
        *head = av_malloc(sizeof(NvencOutputSurfaceList));
        (*head)->next = NULL;
        (*head)->surface = surface;
        return;
    }

    while ((*head)->next)
        head = &((*head)->next);

    (*head)->next = av_malloc(sizeof(NvencOutputSurfaceList));
    (*head)->next->next = NULL;
    (*head)->next->surface = surface;
}

static NvencOutputSurface *out_surf_queue_dequeue(NvencOutputSurfaceList** head)
{
    NvencOutputSurfaceList *tmp;
    NvencOutputSurface *res;

    if (!*head)
        return NULL;

    tmp = *head;
    res = tmp->surface;
    *head = tmp->next;
    av_free(tmp);

    return res;
}

static void timestamp_list_insert_sorted(NvencTimestampList** head, int64_t timestamp)
{
    NvencTimestampList *newelem;
    NvencTimestampList *prev;

    if (!*head) {
        *head = av_malloc(sizeof(NvencTimestampList));
        (*head)->next = NULL;
        (*head)->timestamp = timestamp;
        return;
    }

    prev = NULL;
    while (*head && timestamp >= (*head)->timestamp) {
        prev = *head;
        head = &((*head)->next);
    }

    newelem = av_malloc(sizeof(NvencTimestampList));
    newelem->next = *head;
    newelem->timestamp = timestamp;

    if (*head) {
        *head = newelem;
    } else {
        prev->next = newelem;
    }
}

static int64_t timestamp_list_get_lowest(NvencTimestampList** head)
{
    NvencTimestampList *tmp;
    int64_t res;

    if (!*head)
        return 0;

    tmp = *head;
    res = tmp->timestamp;
    *head = tmp->next;
    av_free(tmp);

    return res;
}

static int nvenc_encode_init(AVCodecContext *avctx)
{
    NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS encode_session_params = { 0 };
    NV_ENC_PRESET_CONFIG preset_config = { 0 };
    CUcontext cu_context_curr;
    CUresult cu_res;
    GUID encoder_preset = NV_ENC_PRESET_HQ_GUID;
    GUID license = dummy_license;
    NVENCSTATUS nv_status = NV_ENC_SUCCESS;
    int surfaceCount = 0;
    int i, numMBs;
    int isLL = 0;

    NvencContext *ctx = avctx->priv_data;

    if (!nvenc_dyload_nvenc(avctx))
        return AVERROR_EXTERNAL;

    avctx->coded_frame = av_frame_alloc();
    if (!avctx->coded_frame)
        return AVERROR(ENOMEM);

    ctx->output_surface_queue = NULL;
    ctx->output_surface_ready_queue = NULL;
    ctx->timestamp_list = NULL;
    ctx->last_dts = AV_NOPTS_VALUE;
    ctx->nvencoder = NULL;

    ctx->encode_config.version = NV_ENC_CONFIG_VER;
    ctx->init_encode_params.version = NV_ENC_INITIALIZE_PARAMS_VER;
    preset_config.version = NV_ENC_PRESET_CONFIG_VER;
    preset_config.presetCfg.version = NV_ENC_CONFIG_VER;
    encode_session_params.version = NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS_VER;
    encode_session_params.apiVersion = NVENCAPI_VERSION;
    encode_session_params.clientKeyPtr = &license;

    ctx->cu_context = NULL;
    cu_res = cu_ctx_create(&ctx->cu_context, 0, nvenc_devices[nvenc_use_device_id]);

    if (cu_res != CUDA_SUCCESS) {
        av_log(avctx, AV_LOG_FATAL, "Failed creating CUDA context for NVENC: 0x%x\n", (int)cu_res);
        goto error;
    }

    cu_res = cu_ctx_pop_current(&cu_context_curr);

    if (cu_res != CUDA_SUCCESS) {
        av_log(avctx, AV_LOG_FATAL, "Failed popping CUDA context: 0x%x\n", (int)cu_res);
        goto error;
    }

    encode_session_params.device = ctx->cu_context;
    encode_session_params.deviceType = NV_ENC_DEVICE_TYPE_CUDA;

    nv_status = p_nvenc->nvEncOpenEncodeSessionEx(&encode_session_params, &ctx->nvencoder);
    if (nv_status != NV_ENC_SUCCESS) {
        ctx->nvencoder = NULL;
        av_log(avctx, AV_LOG_FATAL, "OpenEncodeSessionEx failed: 0x%x - invalid license key?\n", (int)nv_status);
        goto error;
    }

    if (ctx->preset) {
        if (!strcmp(ctx->preset, "hp")) {
            encoder_preset = NV_ENC_PRESET_HP_GUID;
        } else if (!strcmp(ctx->preset, "hq")) {
            encoder_preset = NV_ENC_PRESET_HQ_GUID;
        } else if (!strcmp(ctx->preset, "bd")) {
            encoder_preset = NV_ENC_PRESET_BD_GUID;
        } else if (!strcmp(ctx->preset, "ll")) {
            encoder_preset = NV_ENC_PRESET_LOW_LATENCY_DEFAULT_GUID;
            isLL = 1;
        } else if (!strcmp(ctx->preset, "llhp")) {
            encoder_preset = NV_ENC_PRESET_LOW_LATENCY_HP_GUID;
            isLL = 1;
        } else if (!strcmp(ctx->preset, "llhq")) {
            encoder_preset = NV_ENC_PRESET_LOW_LATENCY_HQ_GUID;
            isLL = 1;
        } else if (!strcmp(ctx->preset, "default")) {
            encoder_preset = NV_ENC_PRESET_DEFAULT_GUID;
        } else {
            av_log(avctx, AV_LOG_FATAL, "Preset \"%s\" is unknown! Supported presets: hp, hq, bd, ll, llhp, llhq, default\n", ctx->preset);
            goto error;
        }
    }

    nv_status = p_nvenc->nvEncGetEncodePresetConfig(ctx->nvencoder, NV_ENC_CODEC_H264_GUID, encoder_preset, &preset_config);
    if (nv_status != NV_ENC_SUCCESS) {
        av_log(avctx, AV_LOG_FATAL, "GetEncodePresetConfig failed: 0x%x\n", (int)nv_status);
        goto error;
    }

    ctx->init_encode_params.encodeGUID = NV_ENC_CODEC_H264_GUID;
    ctx->init_encode_params.encodeHeight = avctx->height;
    ctx->init_encode_params.encodeWidth = avctx->width;
    ctx->init_encode_params.darHeight = avctx->height;
    ctx->init_encode_params.darWidth = avctx->width;
    ctx->init_encode_params.frameRateNum = avctx->time_base.den;
    ctx->init_encode_params.frameRateDen = avctx->time_base.num * avctx->ticks_per_frame;

    numMBs = ((avctx->width + 15) >> 4) * ((avctx->height + 15) >> 4);
    ctx->max_surface_count = (numMBs >= 8160) ? 16 : 32;

    ctx->init_encode_params.enableEncodeAsync = 0;
    ctx->init_encode_params.enablePTD = 1;

    ctx->init_encode_params.presetGUID = encoder_preset;

    ctx->init_encode_params.encodeConfig = &ctx->encode_config;
    memcpy(&ctx->encode_config, &preset_config.presetCfg, sizeof(ctx->encode_config));
    ctx->encode_config.version = NV_ENC_CONFIG_VER;

    if (avctx->gop_size >= 0) {
        ctx->encode_config.gopLength = avctx->gop_size;
        ctx->encode_config.encodeCodecConfig.h264Config.idrPeriod = avctx->gop_size;
    }

    if (avctx->bit_rate > 0)
        ctx->encode_config.rcParams.averageBitRate = avctx->bit_rate;

    if (avctx->rc_max_rate > 0)
        ctx->encode_config.rcParams.maxBitRate = avctx->rc_max_rate;

    if (ctx->cbr) {
        if (!ctx->twopass) {
            ctx->encode_config.rcParams.rateControlMode = NV_ENC_PARAMS_RC_CBR;
        } else if (ctx->twopass == 1 || isLL) {
            ctx->encode_config.rcParams.rateControlMode = NV_ENC_PARAMS_RC_2_PASS_QUALITY;

            ctx->encode_config.encodeCodecConfig.h264Config.adaptiveTransformMode = NV_ENC_H264_ADAPTIVE_TRANSFORM_ENABLE;
            ctx->encode_config.encodeCodecConfig.h264Config.fmoMode = NV_ENC_H264_FMO_DISABLE;

            if (!isLL)
                av_log(avctx, AV_LOG_WARNING, "Twopass mode is only known to work with low latency (ll, llhq, llhp) presets.\n");
        } else {
            ctx->encode_config.rcParams.rateControlMode = NV_ENC_PARAMS_RC_CBR;
        }
    } else if (avctx->global_quality > 0) {
        ctx->encode_config.rcParams.rateControlMode = NV_ENC_PARAMS_RC_CONSTQP;
        ctx->encode_config.rcParams.constQP.qpInterB = avctx->global_quality;
        ctx->encode_config.rcParams.constQP.qpInterP = avctx->global_quality;
        ctx->encode_config.rcParams.constQP.qpIntra = avctx->global_quality;

        avctx->qmin = -1;
        avctx->qmax = -1;
    } else if (avctx->qmin >= 0 && avctx->qmax >= 0) {
        ctx->encode_config.rcParams.rateControlMode = NV_ENC_PARAMS_RC_VBR;

        ctx->encode_config.rcParams.enableMinQP = 1;
        ctx->encode_config.rcParams.enableMaxQP = 1;

        ctx->encode_config.rcParams.minQP.qpInterB = avctx->qmin;
        ctx->encode_config.rcParams.minQP.qpInterP = avctx->qmin;
        ctx->encode_config.rcParams.minQP.qpIntra = avctx->qmin;

        ctx->encode_config.rcParams.maxQP.qpInterB = avctx->qmax;
        ctx->encode_config.rcParams.maxQP.qpInterP = avctx->qmax;
        ctx->encode_config.rcParams.maxQP.qpIntra = avctx->qmax;
    }

    if (avctx->rc_buffer_size > 0)
        ctx->encode_config.rcParams.vbvBufferSize = avctx->rc_buffer_size;

    if (avctx->flags & CODEC_FLAG_INTERLACED_DCT) {
        ctx->encode_config.frameFieldMode = NV_ENC_PARAMS_FRAME_FIELD_MODE_FIELD;
    } else {
        ctx->encode_config.frameFieldMode = NV_ENC_PARAMS_FRAME_FIELD_MODE_FRAME;
    }

    switch (avctx->profile) {
    case FF_PROFILE_H264_BASELINE:
        ctx->encode_config.profileGUID = NV_ENC_H264_PROFILE_BASELINE_GUID;
        break;
    case FF_PROFILE_H264_MAIN:
        ctx->encode_config.profileGUID = NV_ENC_H264_PROFILE_MAIN_GUID;
        break;
    case FF_PROFILE_H264_HIGH:
        ctx->encode_config.profileGUID = NV_ENC_H264_PROFILE_HIGH_GUID;
        break;
    default:
        av_log(avctx, AV_LOG_WARNING, "Unsupported h264 profile requested, falling back to high\n");
        ctx->encode_config.profileGUID = NV_ENC_H264_PROFILE_HIGH_GUID;
        break;
    }

    if (ctx->gobpattern >= 0) {
        ctx->encode_config.frameIntervalP = 1;
    }

    ctx->encode_config.encodeCodecConfig.h264Config.h264VUIParameters.colourDescriptionPresentFlag = 1;
    ctx->encode_config.encodeCodecConfig.h264Config.h264VUIParameters.videoSignalTypePresentFlag = 1;

    ctx->encode_config.encodeCodecConfig.h264Config.h264VUIParameters.colourMatrix = avctx->colorspace;
    ctx->encode_config.encodeCodecConfig.h264Config.h264VUIParameters.colourPrimaries = avctx->color_primaries;
    ctx->encode_config.encodeCodecConfig.h264Config.h264VUIParameters.transferCharacteristics = avctx->color_trc;

    ctx->encode_config.encodeCodecConfig.h264Config.h264VUIParameters.videoFullRangeFlag = avctx->color_range == AVCOL_RANGE_JPEG;

    ctx->encode_config.encodeCodecConfig.h264Config.disableSPSPPS = (avctx->flags & CODEC_FLAG_GLOBAL_HEADER) ? 1 : 0;

    nv_status = p_nvenc->nvEncInitializeEncoder(ctx->nvencoder, &ctx->init_encode_params);
    if (nv_status != NV_ENC_SUCCESS) {
        av_log(avctx, AV_LOG_FATAL, "InitializeEncoder failed: 0x%x\n", (int)nv_status);
        goto error;
    }

    ctx->input_surfaces = av_malloc(ctx->max_surface_count * sizeof(*ctx->input_surfaces));
    ctx->output_surfaces = av_malloc(ctx->max_surface_count * sizeof(*ctx->output_surfaces));

    for (surfaceCount = 0; surfaceCount < ctx->max_surface_count; ++surfaceCount) {
        NV_ENC_CREATE_INPUT_BUFFER allocSurf = { 0 };
        NV_ENC_CREATE_BITSTREAM_BUFFER allocOut = { 0 };
        allocSurf.version = NV_ENC_CREATE_INPUT_BUFFER_VER;
        allocOut.version = NV_ENC_CREATE_BITSTREAM_BUFFER_VER;

        allocSurf.width = (avctx->width + 31) & ~31;
        allocSurf.height = (avctx->height + 31) & ~31;

        allocSurf.memoryHeap = NV_ENC_MEMORY_HEAP_SYSMEM_CACHED;

        switch (avctx->pix_fmt) {
        case AV_PIX_FMT_YUV420P:
            allocSurf.bufferFmt = NV_ENC_BUFFER_FORMAT_YV12_PL;
            break;

        case AV_PIX_FMT_NV12:
            allocSurf.bufferFmt = NV_ENC_BUFFER_FORMAT_NV12_PL;
            break;

            case AV_PIX_FMT_YUV444P:
        allocSurf.bufferFmt = NV_ENC_BUFFER_FORMAT_YUV444_PL;
            break;

        default:
            av_log(avctx, AV_LOG_FATAL, "Invalid input pixel format\n");
            goto error;
        }

        nv_status = p_nvenc->nvEncCreateInputBuffer(ctx->nvencoder, &allocSurf);
        if (nv_status = NV_ENC_SUCCESS){
            av_log(avctx, AV_LOG_FATAL, "CreateInputBuffer failed\n");
            goto error;
        }

        ctx->input_surfaces[surfaceCount].lockCount = 0;
        ctx->input_surfaces[surfaceCount].input_surface = allocSurf.inputBuffer;
        ctx->input_surfaces[surfaceCount].format = allocSurf.bufferFmt;
        ctx->input_surfaces[surfaceCount].width = allocSurf.width;
        ctx->input_surfaces[surfaceCount].height = allocSurf.height;

        /* 1MB is large enough to hold most output frames.
           NVENC increases this automaticaly if it's not enough. */
        allocOut.size = 1024 * 1024;

        allocOut.memoryHeap = NV_ENC_MEMORY_HEAP_SYSMEM_CACHED;

        nv_status = p_nvenc->nvEncCreateBitstreamBuffer(ctx->nvencoder, &allocOut);
        if (nv_status = NV_ENC_SUCCESS) {
            av_log(avctx, AV_LOG_FATAL, "CreateBitstreamBuffer failed\n");
            ctx->output_surfaces[surfaceCount++].output_surface = NULL;
            goto error;
        }

        ctx->output_surfaces[surfaceCount].output_surface = allocOut.bitstreamBuffer;
        ctx->output_surfaces[surfaceCount].size = allocOut.size;
        ctx->output_surfaces[surfaceCount].busy = 0;
    }

    if (avctx->flags & CODEC_FLAG_GLOBAL_HEADER) {
        uint32_t outSize = 0;
        char tmpHeader[256];
        NV_ENC_SEQUENCE_PARAM_PAYLOAD payload = { 0 };
        payload.version = NV_ENC_SEQUENCE_PARAM_PAYLOAD_VER;

        payload.spsppsBuffer = tmpHeader;
        payload.inBufferSize = sizeof(tmpHeader);
        payload.outSPSPPSPayloadSize = &outSize;

        nv_status = p_nvenc->nvEncGetSequenceParams(ctx->nvencoder, &payload);
        if (nv_status != NV_ENC_SUCCESS) {
            av_log(avctx, AV_LOG_FATAL, "GetSequenceParams failed\n");
            goto error;
        }

        avctx->extradata_size = outSize;
        avctx->extradata = av_mallocz(outSize + FF_INPUT_BUFFER_PADDING_SIZE);

        memcpy(avctx->extradata, tmpHeader, outSize);
    }

    if (ctx->encode_config.frameIntervalP > 1)
        avctx->has_b_frames = 2;

    if (ctx->encode_config.rcParams.averageBitRate > 0)
        avctx->bit_rate = ctx->encode_config.rcParams.averageBitRate;

    return 0;

error:

    for (i = 0; i < surfaceCount; ++i) {
        p_nvenc->nvEncDestroyInputBuffer(ctx->nvencoder, ctx->input_surfaces[i].input_surface);
        if (ctx->output_surfaces[i].output_surface)
            p_nvenc->nvEncDestroyBitstreamBuffer(ctx->nvencoder, ctx->output_surfaces[i].output_surface);
    }

    if (ctx->nvencoder)
        p_nvenc->nvEncDestroyEncoder(ctx->nvencoder);

    if (ctx->cu_context)
        cu_ctx_destroy(ctx->cu_context);

    nvenc_unload_nvenc(avctx);

    ctx->nvencoder = NULL;
    ctx->cu_context = NULL;

    return AVERROR_EXTERNAL;
}

static av_cold int nvenc_encode_close(AVCodecContext *avctx)
{
    NvencContext *ctx = avctx->priv_data;
    int i;

    while (ctx->timestamp_list)
        timestamp_list_get_lowest(&ctx->timestamp_list);

    while (ctx->output_surface_ready_queue)
        out_surf_queue_dequeue(&ctx->output_surface_ready_queue);

    while (ctx->output_surface_queue)
        out_surf_queue_dequeue(&ctx->output_surface_queue);

    for (i = 0; i < ctx->max_surface_count; ++i) {
        p_nvenc->nvEncDestroyInputBuffer(ctx->nvencoder, ctx->input_surfaces[i].input_surface);
        p_nvenc->nvEncDestroyBitstreamBuffer(ctx->nvencoder, ctx->output_surfaces[i].output_surface);
    }
    ctx->max_surface_count = 0;

    p_nvenc->nvEncDestroyEncoder(ctx->nvencoder);
    ctx->nvencoder = NULL;

    cu_ctx_destroy(ctx->cu_context);
    ctx->cu_context = NULL;

    nvenc_unload_nvenc(avctx);

    av_frame_free(&avctx->coded_frame);

    return 0;
}

static int process_output_surface(AVCodecContext *avctx, AVPacket *pkt, AVFrame *coded_frame, NvencOutputSurface *tmpoutsurf)
{
    NvencContext *ctx = avctx->priv_data;
    uint32_t *slice_offsets = av_mallocz(ctx->encode_config.encodeCodecConfig.h264Config.sliceModeData * sizeof(*slice_offsets));
    NV_ENC_LOCK_BITSTREAM lock_params = { 0 };
    NVENCSTATUS nv_status;
    int res = 0;

    lock_params.version = NV_ENC_LOCK_BITSTREAM_VER;

    lock_params.doNotWait = 0;
    lock_params.outputBitstream = tmpoutsurf->output_surface;
    lock_params.sliceOffsets = slice_offsets;

    nv_status = p_nvenc->nvEncLockBitstream(ctx->nvencoder, &lock_params);
    if (nv_status != NV_ENC_SUCCESS) {
        av_log(avctx, AV_LOG_ERROR, "Failed locking bitstream buffer\n");
        res = AVERROR_EXTERNAL;
        goto error;
    }

    if (res = ff_alloc_packet2(avctx, pkt, lock_params.bitstreamSizeInBytes)) {
        p_nvenc->nvEncUnlockBitstream(ctx->nvencoder, tmpoutsurf->output_surface);
        goto error;
    }

    memcpy(pkt->data, lock_params.bitstreamBufferPtr, lock_params.bitstreamSizeInBytes);

    nv_status = p_nvenc->nvEncUnlockBitstream(ctx->nvencoder, tmpoutsurf->output_surface);
    if (nv_status != NV_ENC_SUCCESS)
        av_log(avctx, AV_LOG_ERROR, "Failed unlocking bitstream buffer, expect the gates of mordor to open\n");

    switch (lock_params.pictureType) {
    case NV_ENC_PIC_TYPE_IDR:
        pkt->flags |= AV_PKT_FLAG_KEY;
    case NV_ENC_PIC_TYPE_I:
        avctx->coded_frame->pict_type = AV_PICTURE_TYPE_I;
        break;
    case NV_ENC_PIC_TYPE_P:
        avctx->coded_frame->pict_type = AV_PICTURE_TYPE_P;
        break;
    case NV_ENC_PIC_TYPE_B:
        avctx->coded_frame->pict_type = AV_PICTURE_TYPE_B;
        break;
    case NV_ENC_PIC_TYPE_BI:
        avctx->coded_frame->pict_type = AV_PICTURE_TYPE_BI;
        break;
    default:
        avctx->coded_frame->pict_type = AV_PICTURE_TYPE_NONE;
        break;
    }

    pkt->pts = lock_params.outputTimeStamp;
    pkt->dts = timestamp_list_get_lowest(&ctx->timestamp_list);

    if (pkt->dts > pkt->pts)
        pkt->dts = pkt->pts;

    if (ctx->last_dts != AV_NOPTS_VALUE && pkt->dts <= ctx->last_dts)
        pkt->dts = ctx->last_dts + 1;

    ctx->last_dts = pkt->dts;

    av_free(slice_offsets);

    return res;

error:

    av_free(slice_offsets);
    timestamp_list_get_lowest(&ctx->timestamp_list);

    return res;
}

static int nvenc_encode_frame(AVCodecContext *avctx, AVPacket *pkt,
    const AVFrame *frame, int *got_packet)
{
    NVENCSTATUS nv_status;
    NvencContext *ctx = avctx->priv_data;
    NvencOutputSurface *tmpoutsurf;
    int i = 0;

    NV_ENC_PIC_PARAMS pic_params = { 0 };
    pic_params.version = NV_ENC_PIC_PARAMS_VER;

    if (frame) {
        NV_ENC_LOCK_INPUT_BUFFER lockBufferParams = { 0 };
        NvencInputSurface *inSurf = NULL;

        for (i = 0; i < ctx->max_surface_count; ++i)
        {
            if (!ctx->input_surfaces[i].lockCount)
            {
                inSurf = &ctx->input_surfaces[i];
                break;
            }
        }

        av_assert0(inSurf);

        inSurf->lockCount = 1;

        lockBufferParams.version = NV_ENC_LOCK_INPUT_BUFFER_VER;
        lockBufferParams.inputBuffer = inSurf->input_surface;

        nv_status = p_nvenc->nvEncLockInputBuffer(ctx->nvencoder, &lockBufferParams);
        if (nv_status != NV_ENC_SUCCESS) {
            av_log(avctx, AV_LOG_ERROR, "Failed locking nvenc input buffer\n");
            return 0;
        }

        if (avctx->pix_fmt == AV_PIX_FMT_YUV420P) {
            uint8_t *buf = lockBufferParams.bufferDataPtr;

            av_image_copy_plane(buf, lockBufferParams.pitch,
                frame->data[0], frame->linesize[0],
                avctx->width, avctx->height);

            buf += inSurf->height * lockBufferParams.pitch;

            av_image_copy_plane(buf, lockBufferParams.pitch >> 1,
                frame->data[2], frame->linesize[2],
                avctx->width >> 1, avctx->height >> 1);

            buf += (inSurf->height * lockBufferParams.pitch) >> 2;

            av_image_copy_plane(buf, lockBufferParams.pitch >> 1,
                frame->data[1], frame->linesize[1],
                avctx->width >> 1, avctx->height >> 1);
        } else if (avctx->pix_fmt == AV_PIX_FMT_NV12) {
            uint8_t *buf = lockBufferParams.bufferDataPtr;

            av_image_copy_plane(buf, lockBufferParams.pitch,
                frame->data[0], frame->linesize[0],
                avctx->width, avctx->height);

            buf += inSurf->height * lockBufferParams.pitch;

            av_image_copy_plane(buf, lockBufferParams.pitch,
                frame->data[1], frame->linesize[1],
                avctx->width, avctx->height >> 1);
        } else if (avctx->pix_fmt == AV_PIX_FMT_YUV444P) {
            uint8_t *buf = lockBufferParams.bufferDataPtr;

            av_image_copy_plane(buf, lockBufferParams.pitch,
                frame->data[0], frame->linesize[0],
                avctx->width, avctx->height);

            buf += inSurf->height * lockBufferParams.pitch;

            av_image_copy_plane(buf, lockBufferParams.pitch,
                frame->data[1], frame->linesize[1],
                avctx->width, avctx->height);

            buf += inSurf->height * lockBufferParams.pitch;

            av_image_copy_plane(buf, lockBufferParams.pitch,
                frame->data[2], frame->linesize[2],
                avctx->width, avctx->height);
        } else {
            av_log(avctx, AV_LOG_FATAL, "Invalid pixel format!\n");
            return AVERROR(EINVAL);
        }

        nv_status = p_nvenc->nvEncUnlockInputBuffer(ctx->nvencoder, inSurf->input_surface);
        if (nv_status != NV_ENC_SUCCESS) {
            av_log(avctx, AV_LOG_FATAL, "Failed unlocking input buffer!\n");
            return AVERROR_EXTERNAL;
        }

        for (i = 0; i < ctx->max_surface_count; ++i)
            if (!ctx->output_surfaces[i].busy)
                break;

        if (i == ctx->max_surface_count) {
            inSurf->lockCount = 0;
            av_log(avctx, AV_LOG_FATAL, "No free output surface found!\n");
            return AVERROR_EXTERNAL;
        }

        ctx->output_surfaces[i].input_surface = inSurf;

        pic_params.inputBuffer = inSurf->input_surface;
        pic_params.bufferFmt = inSurf->format;
        pic_params.inputWidth = avctx->width;
        pic_params.inputHeight = avctx->height;
        pic_params.outputBitstream = ctx->output_surfaces[i].output_surface;
        pic_params.completionEvent = 0;

        if (avctx->flags & CODEC_FLAG_INTERLACED_DCT) {
            if (frame->top_field_first) {
                pic_params.pictureStruct = NV_ENC_PIC_STRUCT_FIELD_TOP_BOTTOM;
            } else {
                pic_params.pictureStruct = NV_ENC_PIC_STRUCT_FIELD_BOTTOM_TOP;
            }
        } else {
            pic_params.pictureStruct = NV_ENC_PIC_STRUCT_FRAME;
        }

        pic_params.encodePicFlags = 0;
        pic_params.inputTimeStamp = frame->pts;
        pic_params.inputDuration = 0;
        pic_params.codecPicParams.h264PicParams.sliceMode = ctx->encode_config.encodeCodecConfig.h264Config.sliceMode;
        pic_params.codecPicParams.h264PicParams.sliceModeData = ctx->encode_config.encodeCodecConfig.h264Config.sliceModeData;
        memcpy(&pic_params.rcParams, &ctx->encode_config.rcParams, sizeof(NV_ENC_RC_PARAMS));

        timestamp_list_insert_sorted(&ctx->timestamp_list, frame->pts);
    } else {
        pic_params.encodePicFlags = NV_ENC_PIC_FLAG_EOS;
    }

    nv_status = p_nvenc->nvEncEncodePicture(ctx->nvencoder, &pic_params);

    if (frame && nv_status == NV_ENC_ERR_NEED_MORE_INPUT) {
        out_surf_queue_enqueue(&ctx->output_surface_queue, &ctx->output_surfaces[i]);
        ctx->output_surfaces[i].busy = 1;
    }

    if (nv_status != NV_ENC_SUCCESS && nv_status != NV_ENC_ERR_NEED_MORE_INPUT) {
        av_log(avctx, AV_LOG_ERROR, "EncodePicture failed!\n");
        return AVERROR_EXTERNAL;
    }

    if (nv_status != NV_ENC_ERR_NEED_MORE_INPUT) {
        while (ctx->output_surface_queue) {
            tmpoutsurf = out_surf_queue_dequeue(&ctx->output_surface_queue);
            out_surf_queue_enqueue(&ctx->output_surface_ready_queue, tmpoutsurf);
        }

        if (frame) {
            out_surf_queue_enqueue(&ctx->output_surface_ready_queue, &ctx->output_surfaces[i]);
            ctx->output_surfaces[i].busy = 1;
        }
    }

    if (ctx->output_surface_ready_queue) {
        tmpoutsurf = out_surf_queue_dequeue(&ctx->output_surface_ready_queue);

        i = process_output_surface(avctx, pkt, avctx->coded_frame, tmpoutsurf);

        if (i)
            return i;

        tmpoutsurf->busy = 0;
        av_assert0(tmpoutsurf->input_surface->lockCount);
        tmpoutsurf->input_surface->lockCount--;

        *got_packet = 1;
    } else {
        *got_packet = 0;
    }

    return 0;
}

static int pix_fmts_nvenc_initialized;

static enum AVPixelFormat pix_fmts_nvenc[] = {
    AV_PIX_FMT_NV12,
    AV_PIX_FMT_NONE,
    AV_PIX_FMT_NONE,
    AV_PIX_FMT_NONE
};

static av_cold void nvenc_init_static(AVCodec *codec)
{
    NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS encode_session_params = { 0 };
    CUcontext cuctxcur = 0, cuctx = 0;
    NVENCSTATUS nv_status;
    void *nvencoder = 0;
    GUID encodeGuid = NV_ENC_CODEC_H264_GUID;
    GUID license = dummy_license;
    int i = 0, pos = 0;
    int gotnv12 = 0, got420 = 0, got444 = 0;
    uint32_t inputFmtCount = 32;
    NV_ENC_BUFFER_FORMAT inputFmts[32];

    for (i = 0; i < 32; ++i)
        inputFmts[i] = (NV_ENC_BUFFER_FORMAT)0;
    i = 0;

    if (pix_fmts_nvenc_initialized) {
        codec->pix_fmts = pix_fmts_nvenc;
        return;
    }

    if (!nvenc_dyload_nvenc(0)) {
        pix_fmts_nvenc_initialized = 1;
        return;
    }

    encode_session_params.version = NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS_VER;
    encode_session_params.apiVersion = NVENCAPI_VERSION;
    encode_session_params.clientKeyPtr = &license;

    cuctx = 0;
    if (cu_ctx_create(&cuctx, 0, nvenc_devices[nvenc_use_device_id]) != CUDA_SUCCESS) {
        cuctx = 0;
        goto error;
    }

    if (cu_ctx_pop_current(&cuctxcur) != CUDA_SUCCESS)
        goto error;

    encode_session_params.device = (void*)cuctx;
    encode_session_params.deviceType = NV_ENC_DEVICE_TYPE_CUDA;

    nv_status = p_nvenc->nvEncOpenEncodeSessionEx(&encode_session_params, &nvencoder);
    if (nv_status != NV_ENC_SUCCESS) {
        nvencoder = 0;
        goto error;
    }

    nv_status = p_nvenc->nvEncGetInputFormats(nvencoder, encodeGuid, inputFmts, 32, &inputFmtCount);
    if (nv_status != NV_ENC_SUCCESS)
        goto error;

    pos = 0;
    for (i = 0; i < inputFmtCount && pos < 3; ++i) {
        if (!gotnv12 && (inputFmts[i] == NV_ENC_BUFFER_FORMAT_NV12_PL
                || inputFmts[i] == NV_ENC_BUFFER_FORMAT_NV12_TILED16x16
                || inputFmts[i] == NV_ENC_BUFFER_FORMAT_NV12_TILED64x16)) {

            pix_fmts_nvenc[pos++] = AV_PIX_FMT_NV12;
            gotnv12 = 1;
        } else if (!got420 && (inputFmts[i] == NV_ENC_BUFFER_FORMAT_YV12_PL
                || inputFmts[i] == NV_ENC_BUFFER_FORMAT_YV12_TILED16x16
                || inputFmts[i] == NV_ENC_BUFFER_FORMAT_YV12_TILED64x16)) {

            pix_fmts_nvenc[pos++] = AV_PIX_FMT_YUV420P;
            got420 = 1;
        } else if (!got444 && (inputFmts[i] == NV_ENC_BUFFER_FORMAT_YUV444_PL
                || inputFmts[i] == NV_ENC_BUFFER_FORMAT_YUV444_TILED16x16
                || inputFmts[i] == NV_ENC_BUFFER_FORMAT_YUV444_TILED64x16)) {

            pix_fmts_nvenc[pos++] = AV_PIX_FMT_YUV444P;
            got444 = 1;
        }
    }

    pix_fmts_nvenc[pos] = AV_PIX_FMT_NONE;

    pix_fmts_nvenc_initialized = 1;
    codec->pix_fmts = pix_fmts_nvenc;

    p_nvenc->nvEncDestroyEncoder(nvencoder);
    cu_ctx_destroy(cuctx);

    nvenc_unload_nvenc(0);

    return;

error:

    if (nvencoder)
        p_nvenc->nvEncDestroyEncoder(nvencoder);

    if (cuctx)
        cu_ctx_destroy(cuctx);

    pix_fmts_nvenc_initialized = 1;
    pix_fmts_nvenc[0] = AV_PIX_FMT_NV12;
    pix_fmts_nvenc[1] = AV_PIX_FMT_NONE;

    codec->pix_fmts = pix_fmts_nvenc;

    nvenc_unload_nvenc(0);
}

#define OFFSET(x) offsetof(NvencContext, x)
#define VE AV_OPT_FLAG_VIDEO_PARAM | AV_OPT_FLAG_ENCODING_PARAM
static const AVOption options[] = {
    { "preset", "Set the encoding preset (one of hq, hp, bd, ll, llhq, llhp, default)", OFFSET(preset), AV_OPT_TYPE_STRING, { .str = "hq" }, 0, 0, VE },
    { "cbr", "Use cbr encoding mode", OFFSET(cbr), AV_OPT_TYPE_INT, { .i64 = 0 }, 0, 1, VE },
    { "2pass", "Use 2pass cbr encoding mode (low latency mode only)", OFFSET(twopass), AV_OPT_TYPE_INT, { .i64 = -1 }, -1, 1, VE },
    { "goppattern", "Specifies the GOP pattern as follows: 0: I, 1: IPP, 2: IBP, 3: IBBP", OFFSET(gobpattern), AV_OPT_TYPE_INT, { .i64 = -1 }, -1, 3, VE },
    { NULL }
};

static const AVClass nvenc_class = {
    .class_name = "nvenc",
    .item_name = av_default_item_name,
    .option = options,
    .version = LIBAVUTIL_VERSION_INT,
};

static const AVCodecDefault nvenc_defaults[] = {
    { "b", "0" },
    { "qmin", "-1" },
    { "qmax", "-1" },
    { "qdiff", "-1" },
    { "qblur", "-1" },
    { "qcomp", "-1" },
    { NULL },
};

AVCodec ff_nvenc_encoder = {
    .name = "nvenc",
    .long_name = NULL_IF_CONFIG_SMALL("Nvidia NVENC h264 encoder"),
    .type = AVMEDIA_TYPE_VIDEO,
    .id = AV_CODEC_ID_H264,
    .priv_data_size = sizeof(NvencContext),
    .init = nvenc_encode_init,
    .encode2 = nvenc_encode_frame,
    .close = nvenc_encode_close,
    .capabilities = CODEC_CAP_DELAY,
    .priv_class = &nvenc_class,
    .defaults = nvenc_defaults,
    .init_static_data = nvenc_init_static
};

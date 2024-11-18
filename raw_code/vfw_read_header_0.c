static int vfw_read_header(AVFormatContext *s, AVFormatParameters *ap)

{

    struct vfw_ctx *ctx = s->priv_data;

    AVCodecContext *codec;

    AVStream *st;

    int devnum;

    int bisize;

    BITMAPINFO *bi;

    CAPTUREPARMS cparms;

    DWORD biCompression;

    WORD biBitCount;

    int width;

    int height;

    int ret;



    if(!ap->time_base.den) {

        av_log(s, AV_LOG_ERROR, "A time base must be specified.\n");

        return AVERROR_IO;

    }



    ctx->s = s;



    ctx->hwnd = capCreateCaptureWindow(NULL, 0, 0, 0, 0, 0, HWND_MESSAGE, 0);

    if(!ctx->hwnd) {

        av_log(s, AV_LOG_ERROR, "Could not create capture window.\n");

        return AVERROR_IO;

    }



    /* If atoi fails, devnum==0 and the default device is used */

    devnum = atoi(s->filename);



    ret = SendMessage(ctx->hwnd, WM_CAP_DRIVER_CONNECT, devnum, 0);

    if(!ret) {

        av_log(s, AV_LOG_ERROR, "Could not connect to device.\n");

        DestroyWindow(ctx->hwnd);

        return AVERROR(ENODEV);

    }



    SendMessage(ctx->hwnd, WM_CAP_SET_OVERLAY, 0, 0);

    SendMessage(ctx->hwnd, WM_CAP_SET_PREVIEW, 0, 0);



    ret = SendMessage(ctx->hwnd, WM_CAP_SET_CALLBACK_VIDEOSTREAM, 0,

                      (LPARAM) videostream_cb);

    if(!ret) {

        av_log(s, AV_LOG_ERROR, "Could not set video stream callback.\n");

        goto fail_io;

    }



    SetWindowLongPtr(ctx->hwnd, GWLP_USERDATA, (LONG_PTR) ctx);



    st = av_new_stream(s, 0);

    if(!st) {

        vfw_read_close(s);

        return AVERROR_NOMEM;

    }



    /* Set video format */

    bisize = SendMessage(ctx->hwnd, WM_CAP_GET_VIDEOFORMAT, 0, 0);

    if(!bisize)

        goto fail_io;

    bi = av_malloc(bisize);

    if(!bi) {

        vfw_read_close(s);

        return AVERROR_NOMEM;

    }

    ret = SendMessage(ctx->hwnd, WM_CAP_GET_VIDEOFORMAT, bisize, (LPARAM) bi);

    if(!ret)

        goto fail_bi;



    dump_bih(s, &bi->bmiHeader);



    width  = ap->width  ? ap->width  : bi->bmiHeader.biWidth ;

    height = ap->height ? ap->height : bi->bmiHeader.biHeight;

    bi->bmiHeader.biWidth  = width ;

    bi->bmiHeader.biHeight = height;



#if 0

    /* For testing yet unsupported compressions

     * Copy these values from user-supplied verbose information */

    bi->bmiHeader.biWidth       = 320;

    bi->bmiHeader.biHeight      = 240;

    bi->bmiHeader.biPlanes      = 1;

    bi->bmiHeader.biBitCount    = 12;

    bi->bmiHeader.biCompression = MKTAG('I','4','2','0');

    bi->bmiHeader.biSizeImage   = 115200;

    dump_bih(s, &bi->bmiHeader);

#endif



    ret = SendMessage(ctx->hwnd, WM_CAP_SET_VIDEOFORMAT, bisize, (LPARAM) bi);

    if(!ret) {

        av_log(s, AV_LOG_ERROR, "Could not set Video Format.\n");

        goto fail_bi;

    }



    biCompression = bi->bmiHeader.biCompression;

    biBitCount = bi->bmiHeader.biBitCount;



    av_free(bi);



    /* Set sequence setup */

    ret = SendMessage(ctx->hwnd, WM_CAP_GET_SEQUENCE_SETUP, sizeof(cparms),

                      (LPARAM) &cparms);

    if(!ret)

        goto fail_io;



    dump_captureparms(s, &cparms);



    cparms.fYield = 1; // Spawn a background thread

    cparms.dwRequestMicroSecPerFrame =

                               (ap->time_base.num*1000000) / ap->time_base.den;

    cparms.fAbortLeftMouse = 0;

    cparms.fAbortRightMouse = 0;

    cparms.fCaptureAudio = 0;

    cparms.vKeyAbort = 0;



    ret = SendMessage(ctx->hwnd, WM_CAP_SET_SEQUENCE_SETUP, sizeof(cparms),

                      (LPARAM) &cparms);

    if(!ret)

        goto fail_io;



    codec = st->codec;

    codec->time_base = ap->time_base;

    codec->codec_type = CODEC_TYPE_VIDEO;

    codec->width = width;

    codec->height = height;

    codec->pix_fmt = vfw_pixfmt(biCompression, biBitCount);

    if(codec->pix_fmt == PIX_FMT_NONE) {

        codec->codec_id = vfw_codecid(biCompression);

        if(codec->codec_id == CODEC_ID_NONE) {

            av_log(s, AV_LOG_ERROR, "Unknown compression type. "

                             "Please report verbose (-v 9) debug information.\n");

            vfw_read_close(s);

            return AVERROR_PATCHWELCOME;

        }

        codec->bits_per_coded_sample = biBitCount;

    } else {

        codec->codec_id = CODEC_ID_RAWVIDEO;

        if(biCompression == BI_RGB)

            codec->bits_per_coded_sample = biBitCount;

    }



    av_set_pts_info(st, 32, 1, 1000);



    ctx->mutex = CreateMutex(NULL, 0, NULL);

    if(!ctx->mutex) {

        av_log(s, AV_LOG_ERROR, "Could not create Mutex.\n" );

        goto fail_io;

    }

    ctx->event = CreateEvent(NULL, 1, 0, NULL);

    if(!ctx->event) {

        av_log(s, AV_LOG_ERROR, "Could not create Event.\n" );

        goto fail_io;

    }



    ret = SendMessage(ctx->hwnd, WM_CAP_SEQUENCE_NOFILE, 0, 0);

    if(!ret) {

        av_log(s, AV_LOG_ERROR, "Could not start capture sequence.\n" );

        goto fail_io;

    }



    return 0;



fail_bi:

    av_free(bi);



fail_io:

    vfw_read_close(s);

    return AVERROR_IO;

}

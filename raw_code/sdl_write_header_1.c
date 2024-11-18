static int sdl_write_header(AVFormatContext *s)

{

    SDLContext *sdl = s->priv_data;

    AVStream *st = s->streams[0];

    AVCodecContext *encctx = st->codec;

    AVRational sar, dar; /* sample and display aspect ratios */

    int i, ret;

    int flags = SDL_SWSURFACE | sdl->window_fullscreen ? SDL_FULLSCREEN : 0;



    if (!sdl->window_title)

        sdl->window_title = av_strdup(s->filename);

    if (!sdl->icon_title)

        sdl->icon_title = av_strdup(sdl->window_title);



    if (SDL_WasInit(SDL_INIT_VIDEO)) {

        av_log(s, AV_LOG_ERROR,

               "SDL video subsystem was already inited, aborting\n");

        sdl->sdl_was_already_inited = 1;

        ret = AVERROR(EINVAL);

        goto fail;

    }



    if (SDL_Init(SDL_INIT_VIDEO) != 0) {

        av_log(s, AV_LOG_ERROR, "Unable to initialize SDL: %s\n", SDL_GetError());

        ret = AVERROR(EINVAL);

        goto fail;

    }



    if (   s->nb_streams > 1

        || encctx->codec_type != AVMEDIA_TYPE_VIDEO

        || encctx->codec_id   != AV_CODEC_ID_RAWVIDEO) {

        av_log(s, AV_LOG_ERROR, "Only supports one rawvideo stream\n");

        ret = AVERROR(EINVAL);

        goto fail;

    }



    for (i = 0; sdl_overlay_pix_fmt_map[i].pix_fmt != AV_PIX_FMT_NONE; i++) {

        if (sdl_overlay_pix_fmt_map[i].pix_fmt == encctx->pix_fmt) {

            sdl->overlay_fmt = sdl_overlay_pix_fmt_map[i].overlay_fmt;

            break;

        }

    }



    if (!sdl->overlay_fmt) {

        av_log(s, AV_LOG_ERROR,

               "Unsupported pixel format '%s', choose one of yuv420p, yuyv422, or uyvy422\n",

               av_get_pix_fmt_name(encctx->pix_fmt));

        ret = AVERROR(EINVAL);

        goto fail;

    }



    /* compute overlay width and height from the codec context information */

    sar = st->sample_aspect_ratio.num ? st->sample_aspect_ratio : (AVRational){ 1, 1 };

    dar = av_mul_q(sar, (AVRational){ encctx->width, encctx->height });



    /* we suppose the screen has a 1/1 sample aspect ratio */

    if (sdl->window_width && sdl->window_height) {

        /* fit in the window */

        if (av_cmp_q(dar, (AVRational){ sdl->window_width, sdl->window_height }) > 0) {

            /* fit in width */

            sdl->overlay_width  = sdl->window_width;

            sdl->overlay_height = av_rescale(sdl->overlay_width, dar.den, dar.num);

        } else {

            /* fit in height */

            sdl->overlay_height = sdl->window_height;

            sdl->overlay_width  = av_rescale(sdl->overlay_height, dar.num, dar.den);

        }

    } else {

        if (sar.num > sar.den) {

            sdl->overlay_width  = encctx->width;

            sdl->overlay_height = av_rescale(sdl->overlay_width, dar.den, dar.num);

        } else {

            sdl->overlay_height = encctx->height;

            sdl->overlay_width  = av_rescale(sdl->overlay_height, dar.num, dar.den);

        }

        sdl->window_width  = sdl->overlay_width;

        sdl->window_height = sdl->overlay_height;

    }

    sdl->overlay_x = (sdl->window_width  - sdl->overlay_width ) / 2;

    sdl->overlay_y = (sdl->window_height - sdl->overlay_height) / 2;



    SDL_WM_SetCaption(sdl->window_title, sdl->icon_title);

    sdl->surface = SDL_SetVideoMode(sdl->window_width, sdl->window_height,

                                    24, flags);

    if (!sdl->surface) {

        av_log(s, AV_LOG_ERROR, "Unable to set video mode: %s\n", SDL_GetError());

        ret = AVERROR(EINVAL);

        goto fail;

    }



    sdl->overlay = SDL_CreateYUVOverlay(encctx->width, encctx->height,

                                        sdl->overlay_fmt, sdl->surface);

    if (!sdl->overlay || sdl->overlay->pitches[0] < encctx->width) {

        av_log(s, AV_LOG_ERROR,

               "SDL does not support an overlay with size of %dx%d pixels\n",

               encctx->width, encctx->height);

        ret = AVERROR(EINVAL);

        goto fail;

    }



    av_log(s, AV_LOG_VERBOSE, "w:%d h:%d fmt:%s sar:%d/%d -> w:%d h:%d\n",

           encctx->width, encctx->height, av_get_pix_fmt_name(encctx->pix_fmt), sar.num, sar.den,

           sdl->overlay_width, sdl->overlay_height);

    return 0;



fail:

    sdl_write_trailer(s);

    return ret;

}

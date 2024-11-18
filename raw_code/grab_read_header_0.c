static int grab_read_header(AVFormatContext *s1, AVFormatParameters *ap)

{

    VideoData *s = s1->priv_data;

    AVStream *st;

    int video_fd;

    int desired_palette, desired_depth;

    struct video_tuner tuner;

    struct video_audio audio;

    struct video_picture pict;

    int j;

    int vformat_num = FF_ARRAY_ELEMS(video_formats);



    av_log(s1, AV_LOG_WARNING, "V4L input device is deprecated and will be removed in the next release.");



    if (ap->time_base.den <= 0) {

        av_log(s1, AV_LOG_ERROR, "Wrong time base (%d)\n", ap->time_base.den);

        return -1;

    }

    s->time_base = ap->time_base;



    s->video_win.width = ap->width;

    s->video_win.height = ap->height;



    st = avformat_new_stream(s1, NULL);

    if (!st)

        return AVERROR(ENOMEM);

    avpriv_set_pts_info(st, 64, 1, 1000000); /* 64 bits pts in us */



    video_fd = open(s1->filename, O_RDWR);

    if (video_fd < 0) {

        av_log(s1, AV_LOG_ERROR, "%s: %s\n", s1->filename, strerror(errno));

        goto fail;

    }



    if (ioctl(video_fd, VIDIOCGCAP, &s->video_cap) < 0) {

        av_log(s1, AV_LOG_ERROR, "VIDIOCGCAP: %s\n", strerror(errno));

        goto fail;

    }



    if (!(s->video_cap.type & VID_TYPE_CAPTURE)) {

        av_log(s1, AV_LOG_ERROR, "Fatal: grab device does not handle capture\n");

        goto fail;

    }



    /* no values set, autodetect them */

    if (s->video_win.width <= 0 || s->video_win.height <= 0) {

        if (ioctl(video_fd, VIDIOCGWIN, &s->video_win, sizeof(s->video_win)) < 0) {

            av_log(s1, AV_LOG_ERROR, "VIDIOCGWIN: %s\n", strerror(errno));

            goto fail;

        }

    }



    if(av_image_check_size(s->video_win.width, s->video_win.height, 0, s1) < 0)

        return -1;



    desired_palette = -1;

    desired_depth = -1;

    for (j = 0; j < vformat_num; j++) {

        if (ap->pix_fmt == video_formats[j].pix_fmt) {

            desired_palette = video_formats[j].palette;

            desired_depth = video_formats[j].depth;

            break;

        }

    }



    /* set tv standard */

    if (!ioctl(video_fd, VIDIOCGTUNER, &tuner)) {

        tuner.mode = s->standard;

        ioctl(video_fd, VIDIOCSTUNER, &tuner);

    }



    /* unmute audio */

    audio.audio = 0;

    ioctl(video_fd, VIDIOCGAUDIO, &audio);

    memcpy(&s->audio_saved, &audio, sizeof(audio));

    audio.flags &= ~VIDEO_AUDIO_MUTE;

    ioctl(video_fd, VIDIOCSAUDIO, &audio);



    ioctl(video_fd, VIDIOCGPICT, &pict);

    av_dlog(s1, "v4l: colour=%d hue=%d brightness=%d constrast=%d whiteness=%d\n",

            pict.colour, pict.hue, pict.brightness, pict.contrast, pict.whiteness);

    /* try to choose a suitable video format */

    pict.palette = desired_palette;

    pict.depth= desired_depth;

    if (desired_palette == -1 || ioctl(video_fd, VIDIOCSPICT, &pict) < 0) {

        for (j = 0; j < vformat_num; j++) {

            pict.palette = video_formats[j].palette;

            pict.depth = video_formats[j].depth;

            if (-1 != ioctl(video_fd, VIDIOCSPICT, &pict))

                break;

        }

        if (j >= vformat_num)

            goto fail1;

    }



    if (ioctl(video_fd, VIDIOCGMBUF, &s->gb_buffers) < 0) {

        /* try to use read based access */

        int val;



        s->video_win.x = 0;

        s->video_win.y = 0;

        s->video_win.chromakey = -1;

        s->video_win.flags = 0;



        if (ioctl(video_fd, VIDIOCSWIN, s->video_win) < 0) {

            av_log(s1, AV_LOG_ERROR, "VIDIOCSWIN: %s\n", strerror(errno));

            goto fail;

        }



        s->frame_format = pict.palette;



        val = 1;

        if (ioctl(video_fd, VIDIOCCAPTURE, &val) < 0) {

            av_log(s1, AV_LOG_ERROR, "VIDIOCCAPTURE: %s\n", strerror(errno));

            goto fail;

        }



        s->time_frame = av_gettime() * s->time_base.den / s->time_base.num;

        s->use_mmap = 0;

    } else {

        s->video_buf = mmap(0, s->gb_buffers.size, PROT_READ|PROT_WRITE, MAP_SHARED, video_fd, 0);

        if ((unsigned char*)-1 == s->video_buf) {

            s->video_buf = mmap(0, s->gb_buffers.size, PROT_READ|PROT_WRITE, MAP_PRIVATE, video_fd, 0);

            if ((unsigned char*)-1 == s->video_buf) {

                av_log(s1, AV_LOG_ERROR, "mmap: %s\n", strerror(errno));

                goto fail;

            }

        }

        s->gb_frame = 0;

        s->time_frame = av_gettime() * s->time_base.den / s->time_base.num;



        /* start to grab the first frame */

        s->gb_buf.frame = s->gb_frame % s->gb_buffers.frames;

        s->gb_buf.height = s->video_win.height;

        s->gb_buf.width = s->video_win.width;

        s->gb_buf.format = pict.palette;



        if (ioctl(video_fd, VIDIOCMCAPTURE, &s->gb_buf) < 0) {

            if (errno != EAGAIN) {

            fail1:

                av_log(s1, AV_LOG_ERROR, "VIDIOCMCAPTURE: %s\n", strerror(errno));

            } else {

                av_log(s1, AV_LOG_ERROR, "Fatal: grab device does not receive any video signal\n");

            }

            goto fail;

        }

        for (j = 1; j < s->gb_buffers.frames; j++) {

          s->gb_buf.frame = j;

          ioctl(video_fd, VIDIOCMCAPTURE, &s->gb_buf);

        }

        s->frame_format = s->gb_buf.format;

        s->use_mmap = 1;

    }



    for (j = 0; j < vformat_num; j++) {

        if (s->frame_format == video_formats[j].palette) {

            s->frame_size = s->video_win.width * s->video_win.height * video_formats[j].depth / 8;

            st->codec->pix_fmt = video_formats[j].pix_fmt;

            break;

        }

    }



    if (j >= vformat_num)

        goto fail;



    s->fd = video_fd;



    st->codec->codec_type = AVMEDIA_TYPE_VIDEO;

    st->codec->codec_id = AV_CODEC_ID_RAWVIDEO;

    st->codec->width = s->video_win.width;

    st->codec->height = s->video_win.height;

    st->codec->time_base = s->time_base;

    st->codec->bit_rate = s->frame_size * 1/av_q2d(st->codec->time_base) * 8;



    return 0;

 fail:

    if (video_fd >= 0)

        close(video_fd);

    return AVERROR(EIO);

}

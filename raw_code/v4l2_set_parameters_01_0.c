static int v4l2_set_parameters(AVFormatContext *s1, AVFormatParameters *ap)

{

    struct video_data *s = s1->priv_data;

    struct v4l2_input input;

    struct v4l2_standard standard;

    struct v4l2_streamparm streamparm = { 0 };

    struct v4l2_fract *tpf = &streamparm.parm.capture.timeperframe;

    int i, ret;

    AVRational framerate_q;



    streamparm.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;



    if (s->framerate &&

        (ret = av_parse_video_rate(&framerate_q, s->framerate)) < 0) {

        av_log(s1, AV_LOG_ERROR, "Could not parse framerate '%s'.\n",

               s->framerate);

        return ret;

    }



    /* set tv video input */

    memset (&input, 0, sizeof (input));

    input.index = s->channel;

    if (ioctl(s->fd, VIDIOC_ENUMINPUT, &input) < 0) {

        av_log(s1, AV_LOG_ERROR, "The V4L2 driver ioctl enum input failed:\n");

        return AVERROR(EIO);

    }



    av_log(s1, AV_LOG_DEBUG, "The V4L2 driver set input_id: %d, input: %s\n",

            s->channel, input.name);

    if (ioctl(s->fd, VIDIOC_S_INPUT, &input.index) < 0) {

        av_log(s1, AV_LOG_ERROR,

               "The V4L2 driver ioctl set input(%d) failed\n",

                s->channel);

        return AVERROR(EIO);

    }



    if (s->standard) {

        av_log(s1, AV_LOG_DEBUG, "The V4L2 driver set standard: %s\n",

               s->standard);

        /* set tv standard */

        memset (&standard, 0, sizeof (standard));

        for(i=0;;i++) {

            standard.index = i;

            if (ioctl(s->fd, VIDIOC_ENUMSTD, &standard) < 0) {

                av_log(s1, AV_LOG_ERROR,

                       "The V4L2 driver ioctl set standard(%s) failed\n",

                       s->standard);

                return AVERROR(EIO);

            }



            if (!av_strcasecmp(standard.name, s->standard)) {

                break;

            }

        }



        av_log(s1, AV_LOG_DEBUG,

               "The V4L2 driver set standard: %s, id: %"PRIu64"\n",

               s->standard, (uint64_t)standard.id);

        if (ioctl(s->fd, VIDIOC_S_STD, &standard.id) < 0) {

            av_log(s1, AV_LOG_ERROR,

                   "The V4L2 driver ioctl set standard(%s) failed\n",

                   s->standard);

            return AVERROR(EIO);

        }

    }



    if (framerate_q.num && framerate_q.den) {

        av_log(s1, AV_LOG_DEBUG, "Setting time per frame to %d/%d\n",

               framerate_q.den, framerate_q.num);

        tpf->numerator   = framerate_q.den;

        tpf->denominator = framerate_q.num;



        if (ioctl(s->fd, VIDIOC_S_PARM, &streamparm) != 0) {

            av_log(s1, AV_LOG_ERROR,

                   "ioctl set time per frame(%d/%d) failed\n",

                   framerate_q.den, framerate_q.num);

            return AVERROR(EIO);

        }



        if (framerate_q.num != tpf->denominator ||

            framerate_q.den != tpf->numerator) {

            av_log(s1, AV_LOG_INFO,

                   "The driver changed the time per frame from "

                   "%d/%d to %d/%d\n",

                   framerate_q.den, framerate_q.num,

                   tpf->numerator, tpf->denominator);

        }

    } else {

        if (ioctl(s->fd, VIDIOC_G_PARM, &streamparm) != 0) {

            av_log(s1, AV_LOG_ERROR, "ioctl(VIDIOC_G_PARM): %s\n",

                   strerror(errno));

            return AVERROR(errno);

        }

    }

    s1->streams[0]->codec->time_base.den = tpf->denominator;

    s1->streams[0]->codec->time_base.num = tpf->numerator;



    s->timeout = 100 +

        av_rescale_q(1, s1->streams[0]->codec->time_base,

                        (AVRational){1, 1000});



    return 0;

}

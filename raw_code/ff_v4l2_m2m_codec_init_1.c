int ff_v4l2_m2m_codec_init(AVCodecContext *avctx)

{

    int ret = AVERROR(EINVAL);

    struct dirent *entry;

    char node[PATH_MAX];

    DIR *dirp;



    V4L2m2mContext *s = avctx->priv_data;

    s->avctx = avctx;



    dirp = opendir("/dev");

    if (!dirp)

        return AVERROR(errno);



    for (entry = readdir(dirp); entry; entry = readdir(dirp)) {



        if (strncmp(entry->d_name, "video", 5))

            continue;



        snprintf(node, sizeof(node), "/dev/%s", entry->d_name);

        av_log(s->avctx, AV_LOG_DEBUG, "probing device %s\n", node);

        strncpy(s->devname, node, strlen(node) + 1);

        ret = v4l2_probe_driver(s);

        if (!ret)

                break;

    }



    closedir(dirp);



    if (ret) {

        av_log(s->avctx, AV_LOG_ERROR, "Could not find a valid device\n");

        memset(s->devname, 0, sizeof(s->devname));



        return ret;

    }



    av_log(s->avctx, AV_LOG_INFO, "Using device %s\n", node);



    return v4l2_configure_contexts(s);

}

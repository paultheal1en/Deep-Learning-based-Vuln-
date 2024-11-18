static int oss_open (int in, struct oss_params *req,

                     struct oss_params *obt, int *pfd)

{

    int fd;

    int oflags;

    int mmmmssss;

    audio_buf_info abinfo;

    int fmt, freq, nchannels;

    const char *dspname = in ? conf.devpath_in : conf.devpath_out;

    const char *typ = in ? "ADC" : "DAC";



    /* Kludge needed to have working mmap on Linux */

    oflags = conf.try_mmap ? O_RDWR : (in ? O_RDONLY : O_WRONLY);

    fd = open (dspname, oflags | O_NONBLOCK);

    if (-1 == fd) {

        oss_logerr2 (errno, typ, "Failed to open `%s'\n", dspname);

        return -1;

    }



    freq = req->freq;

    nchannels = req->nchannels;

    fmt = req->fmt;



    if (ioctl (fd, SNDCTL_DSP_SAMPLESIZE, &fmt)) {

        oss_logerr2 (errno, typ, "Failed to set sample size %d\n", req->fmt);

        goto err;

    }



    if (ioctl (fd, SNDCTL_DSP_CHANNELS, &nchannels)) {

        oss_logerr2 (errno, typ, "Failed to set number of channels %d\n",

                     req->nchannels);

        goto err;

    }



    if (ioctl (fd, SNDCTL_DSP_SPEED, &freq)) {

        oss_logerr2 (errno, typ, "Failed to set frequency %d\n", req->freq);

        goto err;

    }



    if (ioctl (fd, SNDCTL_DSP_NONBLOCK, NULL)) {

        oss_logerr2 (errno, typ, "Failed to set non-blocking mode\n");

        goto err;

    }



    mmmmssss = (req->nfrags << 16) | ctz32 (req->fragsize);

    if (ioctl (fd, SNDCTL_DSP_SETFRAGMENT, &mmmmssss)) {

        oss_logerr2 (errno, typ, "Failed to set buffer length (%d, %d)\n",

                     req->nfrags, req->fragsize);

        goto err;

    }



    if (ioctl (fd, in ? SNDCTL_DSP_GETISPACE : SNDCTL_DSP_GETOSPACE, &abinfo)) {

        oss_logerr2 (errno, typ, "Failed to get buffer length\n");

        goto err;

    }



    if (!abinfo.fragstotal || !abinfo.fragsize) {

        AUD_log (AUDIO_CAP, "Returned bogus buffer information(%d, %d) for %s\n",

                 abinfo.fragstotal, abinfo.fragsize, typ);

        goto err;

    }



    obt->fmt = fmt;

    obt->nchannels = nchannels;

    obt->freq = freq;

    obt->nfrags = abinfo.fragstotal;

    obt->fragsize = abinfo.fragsize;

    *pfd = fd;



#ifdef DEBUG_MISMATCHES

    if ((req->fmt != obt->fmt) ||

        (req->nchannels != obt->nchannels) ||

        (req->freq != obt->freq) ||

        (req->fragsize != obt->fragsize) ||

        (req->nfrags != obt->nfrags)) {

        dolog ("Audio parameters mismatch\n");

        oss_dump_info (req, obt);

    }

#endif



#ifdef DEBUG

    oss_dump_info (req, obt);

#endif

    return 0;



 err:

    oss_anal_close (&fd);

    return -1;

}

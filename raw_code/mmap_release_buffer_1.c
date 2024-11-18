static void mmap_release_buffer(AVPacket *pkt)
{
    struct v4l2_buffer buf;
    int res, fd;
    struct buff_data *buf_descriptor = pkt->priv;
    memset(&buf, 0, sizeof(struct v4l2_buffer));
    buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory = V4L2_MEMORY_MMAP;
    buf.index = buf_descriptor->index;
    fd = buf_descriptor->fd;
    av_free(buf_descriptor);
    res = ioctl (fd, VIDIOC_QBUF, &buf);
    if (res < 0) {
        av_log(NULL, AV_LOG_ERROR, "ioctl(VIDIOC_QBUF)\n");
    pkt->data = NULL;
    pkt->size = 0;
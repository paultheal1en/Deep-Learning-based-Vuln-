static int oss_poll_in (HWVoiceIn *hw)

{

    OSSVoiceIn *oss = (OSSVoiceIn *) hw;



    return qemu_set_fd_handler (oss->fd, oss_helper_poll_in, NULL, NULL);

}

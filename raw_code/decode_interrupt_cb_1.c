static int decode_interrupt_cb(void *ctx)

{

    return received_nb_signals > transcode_init_done;

}

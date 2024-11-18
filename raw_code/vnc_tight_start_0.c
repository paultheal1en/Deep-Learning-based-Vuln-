static void vnc_tight_start(VncState *vs)

{

    buffer_reset(&vs->tight);



    // make the output buffer be the zlib buffer, so we can compress it later

    vs->tight_tmp = vs->output;

    vs->output = vs->tight;

}

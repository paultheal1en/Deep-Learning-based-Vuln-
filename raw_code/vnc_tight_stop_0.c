static void vnc_tight_stop(VncState *vs)

{

    // switch back to normal output/zlib buffers

    vs->tight = vs->output;

    vs->output = vs->tight_tmp;

}

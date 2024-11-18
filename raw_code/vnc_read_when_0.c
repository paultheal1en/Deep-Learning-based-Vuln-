static void vnc_read_when(VncState *vs, VncReadEvent *func, size_t expecting)

{

    vs->read_handler = func;

    vs->read_handler_expect = expecting;

}

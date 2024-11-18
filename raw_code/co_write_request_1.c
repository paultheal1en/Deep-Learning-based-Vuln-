static void co_write_request(void *opaque)

{

    BDRVSheepdogState *s = opaque;



    qemu_coroutine_enter(s->co_send, NULL);

}

static int mipsnet_can_receive(void *opaque)

{

    MIPSnetState *s = opaque;



    if (s->busy)

        return 0;

    return !mipsnet_buffer_full(s);

}

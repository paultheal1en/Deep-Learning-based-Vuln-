static int pcnet_can_receive(void *opaque)

{

    PCNetState *s = opaque;

    if (CSR_STOP(s) || CSR_SPND(s))

        return 0;



    if (s->recv_pos > 0)

        return 0;



    return sizeof(s->buffer)-16;

}

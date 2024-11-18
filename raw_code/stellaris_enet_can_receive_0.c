static int stellaris_enet_can_receive(void *opaque)

{

    stellaris_enet_state *s = (stellaris_enet_state *)opaque;



    if ((s->rctl & SE_RCTL_RXEN) == 0)

        return 1;



    return (s->np < 31);

}

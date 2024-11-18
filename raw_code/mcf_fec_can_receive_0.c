static int mcf_fec_can_receive(void *opaque)

{

    mcf_fec_state *s = (mcf_fec_state *)opaque;

    return s->rx_enabled;

}

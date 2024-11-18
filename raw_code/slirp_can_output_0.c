int slirp_can_output(void)

{

    return !slirp_vc || qemu_can_send_packet(slirp_vc);

}

static int slirp_socket_load(QEMUFile *f, struct socket *so)

{

    if (tcp_attach(so) < 0)

        return -ENOMEM;



    so->so_urgc = qemu_get_be32(f);

    so->so_ffamily = qemu_get_be16(f);

    switch (so->so_ffamily) {

    case AF_INET:

        so->so_faddr.s_addr = qemu_get_be32(f);

        so->so_fport = qemu_get_be16(f);

        break;

    default:

        error_report(

                "so_ffamily unknown, unable to restore so_faddr and so_lport\n");

    }

    so->so_lfamily = qemu_get_be16(f);

    switch (so->so_lfamily) {

    case AF_INET:

        so->so_laddr.s_addr = qemu_get_be32(f);

        so->so_lport = qemu_get_be16(f);

        break;

    default:

        error_report(

                "so_ffamily unknown, unable to restore so_laddr and so_lport\n");

    }

    so->so_iptos = qemu_get_byte(f);

    so->so_emu = qemu_get_byte(f);

    so->so_type = qemu_get_byte(f);

    so->so_state = qemu_get_be32(f);

    if (slirp_sbuf_load(f, &so->so_rcv) < 0)

        return -ENOMEM;

    if (slirp_sbuf_load(f, &so->so_snd) < 0)

        return -ENOMEM;

    slirp_tcp_load(f, so->so_tcpcb);



    return 0;

}

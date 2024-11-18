int slirp_remove_hostfwd(int is_udp, struct in_addr host_addr, int host_port)

{

    struct socket *so;

    struct socket *head = (is_udp ? &udb : &tcb);

    struct sockaddr_in addr;

    int port = htons(host_port);

    socklen_t addr_len;

    int n = 0;



 loop_again:

    for (so = head->so_next; so != head; so = so->so_next) {

        addr_len = sizeof(addr);

        if (getsockname(so->s, (struct sockaddr *)&addr, &addr_len) == 0 &&

            addr.sin_addr.s_addr == host_addr.s_addr &&

            addr.sin_port == port) {

            close(so->s);

            sofree(so);

            n++;

            goto loop_again;

        }

    }



    return n;

}

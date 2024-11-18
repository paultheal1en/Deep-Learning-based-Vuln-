int tcp_ctl(struct socket *so)

{

    Slirp *slirp = so->slirp;

    struct sbuf *sb = &so->so_snd;

    struct ex_list *ex_ptr;

    int do_pty;



    DEBUG_CALL("tcp_ctl");

    DEBUG_ARG("so = %lx", (long )so);



    if (so->so_faddr.s_addr != slirp->vhost_addr.s_addr) {

        /* Check if it's pty_exec */

        for (ex_ptr = slirp->exec_list; ex_ptr; ex_ptr = ex_ptr->ex_next) {

            if (ex_ptr->ex_fport == so->so_fport &&

                so->so_faddr.s_addr == ex_ptr->ex_addr.s_addr) {

                if (ex_ptr->ex_pty == 3) {

                    so->s = -1;

                    so->extra = (void *)ex_ptr->ex_exec;

                    return 1;

                }

                do_pty = ex_ptr->ex_pty;

                DEBUG_MISC((dfd, " executing %s \n",ex_ptr->ex_exec));

                return fork_exec(so, ex_ptr->ex_exec, do_pty);

            }

        }

    }

    sb->sb_cc =

        snprintf(sb->sb_wptr, sb->sb_datalen - (sb->sb_wptr - sb->sb_data),

                 "Error: No application configured.\r\n");

    sb->sb_wptr += sb->sb_cc;

    return 0;

}

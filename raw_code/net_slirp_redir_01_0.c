void net_slirp_redir(Monitor *mon, const char *redir_str, const char *redir_opt2)

{

    struct slirp_config_str *config;



    if (!slirp_inited) {

        if (mon) {

            monitor_printf(mon, "user mode network stack not in use\n");

        } else {

            config = qemu_malloc(sizeof(*config));

            config->str = redir_str;

            config->next = slirp_redirs;

            slirp_redirs = config;

        }

        return;

    }



    if (!strcmp(redir_str, "remove")) {

        net_slirp_redir_rm(mon, redir_opt2);

        return;

    }



    slirp_redirection(mon, redir_str);

}

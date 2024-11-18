static int authenticate(BDRVSSHState *s, const char *user)

{

    int r, ret;

    const char *userauthlist;

    LIBSSH2_AGENT *agent = NULL;

    struct libssh2_agent_publickey *identity;

    struct libssh2_agent_publickey *prev_identity = NULL;



    userauthlist = libssh2_userauth_list(s->session, user, strlen(user));

    if (strstr(userauthlist, "publickey") == NULL) {

        ret = -EPERM;

        error_report("remote server does not support \"publickey\" authentication");

        goto out;

    }



    /* Connect to ssh-agent and try each identity in turn. */

    agent = libssh2_agent_init(s->session);

    if (!agent) {

        ret = -EINVAL;

        session_error_report(s, "failed to initialize ssh-agent support");

        goto out;

    }

    if (libssh2_agent_connect(agent)) {

        ret = -ECONNREFUSED;

        session_error_report(s, "failed to connect to ssh-agent");

        goto out;

    }

    if (libssh2_agent_list_identities(agent)) {

        ret = -EINVAL;

        session_error_report(s, "failed requesting identities from ssh-agent");

        goto out;

    }



    for(;;) {

        r = libssh2_agent_get_identity(agent, &identity, prev_identity);

        if (r == 1) {           /* end of list */

            break;

        }

        if (r < 0) {

            ret = -EINVAL;

            session_error_report(s, "failed to obtain identity from ssh-agent");

            goto out;

        }

        r = libssh2_agent_userauth(agent, user, identity);

        if (r == 0) {

            /* Authenticated! */

            ret = 0;

            goto out;

        }

        /* Failed to authenticate with this identity, try the next one. */

        prev_identity = identity;

    }



    ret = -EPERM;

    error_report("failed to authenticate using publickey authentication "

                 "and the identities held by your ssh-agent");



 out:

    if (agent != NULL) {

        /* Note: libssh2 implementation implicitly calls

         * libssh2_agent_disconnect if necessary.

         */

        libssh2_agent_free(agent);

    }



    return ret;

}

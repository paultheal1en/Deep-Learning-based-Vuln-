static av_cold int libssh_authentication(LIBSSHContext *libssh, const char *user, const char *password)

{

    int authorized = 0;

    int auth_methods;



    if (user)

        ssh_options_set(libssh->session, SSH_OPTIONS_USER, user);



    if (ssh_userauth_none(libssh->session, NULL) == SSH_AUTH_SUCCESS)

        return 0;



    auth_methods = ssh_userauth_list(libssh->session, NULL);



    if (auth_methods & SSH_AUTH_METHOD_PUBLICKEY) {

        if (libssh->priv_key) {

            ssh_string pub_key;

            ssh_private_key priv_key;

            int type;

            if (!ssh_try_publickey_from_file(libssh->session, libssh->priv_key, &pub_key, &type)) {

                priv_key = privatekey_from_file(libssh->session, libssh->priv_key, type, password);

                if (ssh_userauth_pubkey(libssh->session, NULL, pub_key, priv_key) == SSH_AUTH_SUCCESS) {

                    av_log(libssh, AV_LOG_DEBUG, "Authentication successful with selected private key.\n");

                    authorized = 1;

                }

            } else {

                av_log(libssh, AV_LOG_DEBUG, "Invalid key is provided.\n");

                return AVERROR(EACCES);

            }

        } else if (ssh_userauth_autopubkey(libssh->session, password) == SSH_AUTH_SUCCESS) {

            av_log(libssh, AV_LOG_DEBUG, "Authentication successful with auto selected key.\n");

            authorized = 1;

        }

    }



    if (!authorized && (auth_methods & SSH_AUTH_METHOD_PASSWORD)) {

        if (ssh_userauth_password(libssh->session, NULL, password) == SSH_AUTH_SUCCESS) {

            av_log(libssh, AV_LOG_DEBUG, "Authentication successful with password.\n");

            authorized = 1;

        }

    }



    if (!authorized) {

        av_log(libssh, AV_LOG_ERROR, "Authentication failed.\n");

        return AVERROR(EACCES);

    }



    return 0;

}

static int libssh_open(URLContext *h, const char *url, int flags)

{

    static const int verbosity = SSH_LOG_NOLOG;

    LIBSSHContext *s = h->priv_data;

    char proto[10], path[MAX_URL_SIZE], hostname[1024], credencials[1024];

    int port = 22, access, ret;

    long timeout = s->rw_timeout * 1000;

    const char *user = NULL, *pass = NULL;

    char *end = NULL;

    sftp_attributes stat;



    av_url_split(proto, sizeof(proto),

                 credencials, sizeof(credencials),

                 hostname, sizeof(hostname),

                 &port,

                 path, sizeof(path),

                 url);



    if (port <= 0 || port > 65535)

        port = 22;



    if (!(s->session = ssh_new())) {

        ret = AVERROR(ENOMEM);

        goto fail;

    }

    user = av_strtok(credencials, ":", &end);

    pass = av_strtok(end, ":", &end);

    ssh_options_set(s->session, SSH_OPTIONS_HOST, hostname);

    ssh_options_set(s->session, SSH_OPTIONS_PORT, &port);

    ssh_options_set(s->session, SSH_OPTIONS_LOG_VERBOSITY, &verbosity);

    if (timeout > 0)

        ssh_options_set(s->session, SSH_OPTIONS_TIMEOUT_USEC, &timeout);

    if (user)

        ssh_options_set(s->session, SSH_OPTIONS_USER, user);



    if (ssh_connect(s->session) != SSH_OK) {

        av_log(h, AV_LOG_ERROR, "Connection failed. %s\n", ssh_get_error(s->session));

        ret = AVERROR(EIO);

        goto fail;

    }



    if (pass && ssh_userauth_password(s->session, NULL, pass) != SSH_AUTH_SUCCESS) {

        av_log(h, AV_LOG_ERROR, "Error authenticating with password: %s\n", ssh_get_error(s->session));

        ret = AVERROR(EACCES);

        goto fail;

    }



    if (!(s->sftp = sftp_new(s->session))) {

        av_log(h, AV_LOG_ERROR, "SFTP session creation failed: %s\n", ssh_get_error(s->session));

        ret = AVERROR(ENOMEM);

        goto fail;

    }



    if (sftp_init(s->sftp) != SSH_OK) {

        av_log(h, AV_LOG_ERROR, "Error initializing sftp session: %s\n", ssh_get_error(s->session));

        ret = AVERROR(EIO);

        goto fail;

    }



    if ((flags & AVIO_FLAG_WRITE) && (flags & AVIO_FLAG_READ)) {

        access = O_CREAT | O_RDWR;

        if (s->trunc)

            access |= O_TRUNC;

    } else if (flags & AVIO_FLAG_WRITE) {

        access = O_CREAT | O_WRONLY;

        if (s->trunc)

            access |= O_TRUNC;

    } else {

        access = O_RDONLY;

    }



    /* 0666 = -rw-rw-rw- = read+write for everyone, minus umask */

    if (!(s->file = sftp_open(s->sftp, path, access, 0666))) {

        av_log(h, AV_LOG_ERROR, "Error opening sftp file: %s\n", ssh_get_error(s->session));

        ret = AVERROR(EIO);

        goto fail;

    }



    if (!(stat = sftp_fstat(s->file))) {

        av_log(h, AV_LOG_WARNING, "Cannot stat remote file %s.\n", path);

        s->filesize = -1;

    } else {

        s->filesize = stat->size;

        sftp_attributes_free(stat);

    }



    return 0;



  fail:

    libssh_close(h);

    return ret;

}

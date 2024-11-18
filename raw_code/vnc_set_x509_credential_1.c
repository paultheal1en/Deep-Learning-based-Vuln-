static int vnc_set_x509_credential(VncDisplay *vd,

                                   const char *certdir,

                                   const char *filename,

                                   char **cred,

                                   int ignoreMissing)

{

    struct stat sb;



    g_free(*cred);

    *cred = g_malloc(strlen(certdir) + strlen(filename) + 2);



    strcpy(*cred, certdir);

    strcat(*cred, "/");

    strcat(*cred, filename);



    VNC_DEBUG("Check %s\n", *cred);

    if (stat(*cred, &sb) < 0) {

        g_free(*cred);

        *cred = NULL;

        if (ignoreMissing && errno == ENOENT)

            return 0;

        return -1;

    }



    return 0;

}

static int tpm_passthrough_open_sysfs_cancel(TPMPassthruState *tpm_pt)

{

    int fd = -1;

    char *dev;

    char path[PATH_MAX];



    if (tpm_pt->options->cancel_path) {

        fd = qemu_open(tpm_pt->options->cancel_path, O_WRONLY);

        if (fd < 0) {

            error_report("Could not open TPM cancel path : %s",

                         strerror(errno));

        }

        return fd;

    }



    dev = strrchr(tpm_pt->tpm_dev, '/');

    if (dev) {

        dev++;

        if (snprintf(path, sizeof(path), "/sys/class/misc/%s/device/cancel",

                     dev) < sizeof(path)) {

            fd = qemu_open(path, O_WRONLY);

            if (fd >= 0) {

                tpm_pt->options->cancel_path = g_strdup(path);

            } else {

                error_report("tpm_passthrough: Could not open TPM cancel "

                             "path %s : %s", path, strerror(errno));

            }

        }

    } else {

       error_report("tpm_passthrough: Bad TPM device path %s",

                    tpm_pt->tpm_dev);

    }



    return fd;

}

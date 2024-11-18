static void xenstore_update_be(char *watch, char *type, int dom,

                               struct XenDevOps *ops)

{

    struct XenDevice *xendev;

    char path[XEN_BUFSIZE], *dom0, *bepath;

    unsigned int len, dev;



    dom0 = xs_get_domain_path(xenstore, 0);

    len = snprintf(path, sizeof(path), "%s/backend/%s/%d", dom0, type, dom);

    free(dom0);

    if (strncmp(path, watch, len) != 0) {

        return;

    }

    if (sscanf(watch+len, "/%u/%255s", &dev, path) != 2) {

        strcpy(path, "");

        if (sscanf(watch+len, "/%u", &dev) != 1) {

            dev = -1;

        }

    }

    if (dev == -1) {

        return;

    }



    xendev = xen_be_get_xendev(type, dom, dev, ops);

    if (xendev != NULL) {

        bepath = xs_read(xenstore, 0, xendev->be, &len);

        if (bepath == NULL) {

            xen_be_del_xendev(dom, dev);

        } else {

            free(bepath);

            xen_be_backend_changed(xendev, path);

            xen_be_check_state(xendev);

        }

    }

}

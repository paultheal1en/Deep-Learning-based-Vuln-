static void *show_parts(void *arg)

{

    char *device = arg;

    int nbd;



    /* linux just needs an open() to trigger

     * the partition table update

     * but remember to load the module with max_part != 0 :

     *     modprobe nbd max_part=63

     */

    nbd = open(device, O_RDWR);

    if (nbd != -1) {

        close(nbd);

    }

    return NULL;

}

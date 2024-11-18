void add_boot_device_path(int32_t bootindex, DeviceState *dev,

                          const char *suffix)

{

    FWBootEntry *node, *i;



    if (bootindex < 0) {

        return;

    }



    assert(dev != NULL || suffix != NULL);



    node = g_malloc0(sizeof(FWBootEntry));

    node->bootindex = bootindex;

    node->suffix = suffix ? g_strdup(suffix) : NULL;

    node->dev = dev;



    QTAILQ_FOREACH(i, &fw_boot_order, link) {

        if (i->bootindex == bootindex) {

            fprintf(stderr, "Two devices with same boot index %d\n", bootindex);

            exit(1);

        } else if (i->bootindex < bootindex) {

            continue;

        }

        QTAILQ_INSERT_BEFORE(i, node, link);

        return;

    }

    QTAILQ_INSERT_TAIL(&fw_boot_order, node, link);

}

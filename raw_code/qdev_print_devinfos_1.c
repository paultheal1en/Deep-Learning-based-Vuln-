static void qdev_print_devinfos(bool show_no_user)

{

    static const char *cat_name[DEVICE_CATEGORY_MAX + 1] = {

        [DEVICE_CATEGORY_BRIDGE]  = "Controller/Bridge/Hub",

        [DEVICE_CATEGORY_USB]     = "USB",

        [DEVICE_CATEGORY_STORAGE] = "Storage",

        [DEVICE_CATEGORY_NETWORK] = "Network",

        [DEVICE_CATEGORY_INPUT]   = "Input",

        [DEVICE_CATEGORY_DISPLAY] = "Display",

        [DEVICE_CATEGORY_SOUND]   = "Sound",

        [DEVICE_CATEGORY_MISC]    = "Misc",

        [DEVICE_CATEGORY_MAX]     = "Uncategorized",

    };

    GSList *list, *elt;

    int i;

    bool cat_printed;



    list = g_slist_sort(object_class_get_list(TYPE_DEVICE, false),

                        devinfo_cmp);



    for (i = 0; i <= DEVICE_CATEGORY_MAX; i++) {

        cat_printed = false;

        for (elt = list; elt; elt = elt->next) {

            DeviceClass *dc = OBJECT_CLASS_CHECK(DeviceClass, elt->data,

                                                 TYPE_DEVICE);

            if ((i < DEVICE_CATEGORY_MAX

                 ? !test_bit(i, dc->categories)

                 : !bitmap_empty(dc->categories, DEVICE_CATEGORY_MAX))

                || (!show_no_user && dc->no_user)) {

                continue;

            }

            if (!cat_printed) {

                error_printf("%s%s devices:\n", i ? "\n" : "",

                             cat_name[i]);

                cat_printed = true;

            }

            qdev_print_devinfo(dc);

        }

    }



    g_slist_free(list);

}

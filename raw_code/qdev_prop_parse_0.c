int qdev_prop_parse(DeviceState *dev, const char *name, const char *value)

{

    Property *prop;

    int ret;



    prop = qdev_prop_find(dev, name);

    if (!prop) {

        fprintf(stderr, "property \"%s.%s\" not found\n",

                dev->info->name, name);

        return -1;

    }

    if (!prop->info->parse) {

        fprintf(stderr, "property \"%s.%s\" has no parser\n",

                dev->info->name, name);

        return -1;

    }

    ret = prop->info->parse(dev, prop, value);

    if (ret < 0) {

        switch (ret) {

        case -EEXIST:

            fprintf(stderr, "property \"%s.%s\": \"%s\" is already in use\n",

                    dev->info->name, name, value);

            break;

        default:

        case -EINVAL:

            fprintf(stderr, "property \"%s.%s\": failed to parse \"%s\"\n",

                    dev->info->name, name, value);

            break;

        case -ENOENT:

            fprintf(stderr, "property \"%s.%s\": could not find \"%s\"\n",

                    dev->info->name, name, value);

            break;

        }

        return -1;

    }

    return 0;

}

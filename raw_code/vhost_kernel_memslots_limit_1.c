static int vhost_kernel_memslots_limit(struct vhost_dev *dev)

{

    int limit = 64;

    char *s;



    if (g_file_get_contents("/sys/module/vhost/parameters/max_mem_regions",

                            &s, NULL, NULL)) {

        uint64_t val = g_ascii_strtoull(s, NULL, 10);

        if (!((val == G_MAXUINT64 || !val) && errno)) {


            return val;

        }

        error_report("ignoring invalid max_mem_regions value in vhost module:"

                     " %s", s);

    }


    return limit;

}
static int get_monitor_def(target_long *pval, const char *name)

{

    const MonitorDef *md;

    void *ptr;



    for(md = monitor_defs; md->name != NULL; md++) {

        if (compare_cmd(name, md->name)) {

            if (md->get_value) {

                *pval = md->get_value(md, md->offset);

            } else {

                CPUState *env = mon_get_cpu();

                if (!env)

                    return -2;

                ptr = (uint8_t *)env + md->offset;

                switch(md->type) {

                case MD_I32:

                    *pval = *(int32_t *)ptr;

                    break;

                case MD_TLONG:

                    *pval = *(target_long *)ptr;

                    break;

                default:

                    *pval = 0;

                    break;

                }

            }

            return 0;

        }

    }

    return -1;

}

static void set_cfg_value(bool is_max, int index, int value)

{

    if (is_max) {

        cfg.buckets[index].max = value;



    } else {

        cfg.buckets[index].avg = value;

    }

}
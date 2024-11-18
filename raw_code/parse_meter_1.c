static int parse_meter(DBEContext *s)

{

    if (s->meter_size)

        skip_input(s, s->key_present + s->meter_size + 1);

    return 0;

}

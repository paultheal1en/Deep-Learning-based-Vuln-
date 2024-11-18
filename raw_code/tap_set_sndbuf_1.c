static void tap_set_sndbuf(TAPState *s, const char *sndbuf_str, Monitor *mon)

{

    if (sndbuf_str) {

        config_error(mon, "No '-net tap,sndbuf=<nbytes>' support available\n");

    }

}

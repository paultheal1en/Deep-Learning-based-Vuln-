static bool e1000_full_mac_needed(void *opaque)

{

    E1000State *s = opaque;



    return s->compat_flags & E1000_FLAG_MAC;

}

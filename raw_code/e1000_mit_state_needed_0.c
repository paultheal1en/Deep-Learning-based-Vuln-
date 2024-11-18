static bool e1000_mit_state_needed(void *opaque)

{

    E1000State *s = opaque;



    return s->compat_flags & E1000_FLAG_MIT;

}

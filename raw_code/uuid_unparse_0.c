void uuid_unparse(const uuid_t uu, char *out)

{

    snprintf(out, 37, UUID_FMT,

            uu[0], uu[1], uu[2], uu[3], uu[4], uu[5], uu[6], uu[7],

            uu[8], uu[9], uu[10], uu[11], uu[12], uu[13], uu[14], uu[15]);

}

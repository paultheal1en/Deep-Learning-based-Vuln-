void add_command(const cmdinfo_t *ci)

{

    cmdtab = realloc((void *)cmdtab, ++ncmds * sizeof(*cmdtab));

    cmdtab[ncmds - 1] = *ci;

    qsort(cmdtab, ncmds, sizeof(*cmdtab), compare);

}

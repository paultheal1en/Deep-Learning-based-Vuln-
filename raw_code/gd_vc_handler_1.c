static CharDriverState *gd_vc_handler(ChardevVC *vc, Error **errp)
{
    ChardevCommon *common = qapi_ChardevVC_base(vc);
    CharDriverState *chr;
    chr = qemu_chr_alloc(common, errp);
    if (!chr) {
    chr->chr_write = gd_vc_chr_write;
    chr->chr_set_echo = gd_vc_chr_set_echo;
    /* Temporary, until gd_vc_vte_init runs.  */
    chr->opaque = g_new0(VirtualConsole, 1);
    vcs[nb_vcs++] = chr;
    return chr;
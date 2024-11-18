static void qemu_gluster_gconf_free(GlusterConf *gconf)

{

    g_free(gconf->server);

    g_free(gconf->volname);

    g_free(gconf->image);

    g_free(gconf->transport);

    g_free(gconf);

}

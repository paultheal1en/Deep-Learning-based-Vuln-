filter_mirror_set_outdev(Object *obj, const char *value, Error **errp)

{

    MirrorState *s = FILTER_MIRROR(obj);



    g_free(s->outdev);

    s->outdev = g_strdup(value);

    if (!s->outdev) {

        error_setg(errp, "filter filter mirror needs 'outdev' "

                   "property set");

        return;

    }

}

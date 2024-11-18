DevicePropertyInfoList *qmp_device_list_properties(const char *typename,
                                                   Error **errp)
{
    ObjectClass *klass;
    Object *obj;
    ObjectProperty *prop;
    DevicePropertyInfoList *prop_list = NULL;
    klass = object_class_by_name(typename);
    if (klass == NULL) {
        error_set(errp, ERROR_CLASS_DEVICE_NOT_FOUND,
                  "Device '%s' not found", typename);
    klass = object_class_dynamic_cast(klass, TYPE_DEVICE);
    if (klass == NULL) {
        error_setg(errp, QERR_INVALID_PARAMETER_VALUE, "name", TYPE_DEVICE);
    if (object_class_is_abstract(klass)) {
        error_setg(errp, QERR_INVALID_PARAMETER_VALUE, "name",
                   "non-abstract device type");
    obj = object_new(typename);
    QTAILQ_FOREACH(prop, &obj->properties, node) {
        DevicePropertyInfo *info;
        DevicePropertyInfoList *entry;
        /* Skip Object and DeviceState properties */
        if (strcmp(prop->name, "type") == 0 ||
            strcmp(prop->name, "realized") == 0 ||
            strcmp(prop->name, "hotpluggable") == 0 ||
            strcmp(prop->name, "hotplugged") == 0 ||
            strcmp(prop->name, "parent_bus") == 0) {
            continue;
        /* Skip legacy properties since they are just string versions of
         * properties that we already list.
         */
        if (strstart(prop->name, "legacy-", NULL)) {
            continue;
        info = make_device_property_info(klass, prop->name, prop->type,
                                         prop->description);
        if (!info) {
            continue;
        entry = g_malloc0(sizeof(*entry));
        entry->value = info;
        entry->next = prop_list;
        prop_list = entry;
    object_unref(obj);
    return prop_list;
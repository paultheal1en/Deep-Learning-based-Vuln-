static uint32_t parse_enumeration(char *str,
    EnumTable *table, uint32_t not_found_value)
{
    uint32_t ret = not_found_value;
    while (table->name != NULL) {
        if (strcmp(table->name, str) == 0) {
            ret = table->value;
            break;
        }
        table++;
    }
    return ret;
}
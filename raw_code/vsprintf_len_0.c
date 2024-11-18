vsprintf_len(string, format, args)

	char *string;

	const char *format;

	va_list args;

{

	vsprintf(string, format, args);

	return strlen(string);

}

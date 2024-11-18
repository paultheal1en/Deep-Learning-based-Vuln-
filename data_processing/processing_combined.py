import os
import sys
import argparse
import json
import clang.cindex
import clang.enumerations
import csv
import numpy as np
import re
from tqdm.notebook import tqdm
import warnings
import nltk
from gensim.models import Word2Vec
warnings.filterwarnings('ignore')
# set the config
try:
    clang.cindex.Config.set_library_path("/usr/lib/x86_64-linux-gnu")
    clang.cindex.Config.set_library_file('/usr/lib/x86_64-linux-gnu/libclang-10.so.1')
except:
    pass
from graphviz import Digraph
import create_ggnn_data
import split_data


l_funcs = ['StrNCat', 'getaddrinfo', '_ui64toa', 'fclose', 'pthread_mutex_lock', 'gets_s', 'sleep', 
           '_ui64tot', 'freopen_s', '_ui64tow', 'send', 'lstrcat', 'HMAC_Update', '__fxstat', 'StrCatBuff', 
           '_mbscat', '_mbstok_s', '_cprintf_s', 'ldap_search_init_page', 'memmove_s', 'ctime_s', 'vswprintf', 
           'vswprintf_s', '_snwprintf', '_gmtime_s', '_tccpy', '*RC6*', '_mbslwr_s', 'random', 
           '__wcstof_internal', '_wcslwr_s', '_ctime32_s', 'wcsncat*', 'MD5_Init', '_ultoa', 
           'snprintf', 'memset', 'syslog', '_vsnprintf_s', 'HeapAlloc', 'pthread_mutex_destroy', 
           'ChangeWindowMessageFilter', '_ultot', 'crypt_r', '_strupr_s_l', 'LoadLibraryExA', '_strerror_s', 
           'LoadLibraryExW', 'wvsprintf', 'MoveFileEx', '_strdate_s', 'SHA1', 'sprintfW', 'StrCatNW', 
           '_scanf_s_l', 'pthread_attr_init', '_wtmpnam_s', 'snscanf', '_sprintf_s_l', 'dlopen', 
           'sprintfA', 'timed_mutex', 'OemToCharA', 'ldap_delete_ext', 'sethostid', 'popen', 'OemToCharW', 
           '_gettws', 'vfork', '_wcsnset_s_l', 'sendmsg', '_mbsncat', 'wvnsprintfA', 'HeapFree', '_wcserror_s', 
           'realloc', '_snprintf*', 'wcstok', '_strncat*', 'StrNCpy', '_wasctime_s', 'push*', '_lfind_s', 
           'CC_SHA512', 'ldap_compare_ext_s', 'wcscat_s', 'strdup', '_chsize_s', 'sprintf_s', 'CC_MD4_Init', 
           'wcsncpy', '_wfreopen_s', '_wcsupr_s', '_searchenv_s', 'ldap_modify_ext_s', '_wsplitpath', 
           'CC_SHA384_Final', 'MD2', 'RtlCopyMemory', 'lstrcatW', 'MD4', 'MD5', '_wcstok_s_l', '_vsnwprintf_s', 
           'ldap_modify_s', 'strerror', '_lsearch_s', '_mbsnbcat_s', '_wsplitpath_s', 'MD4_Update', '_mbccpy_s', 
           '_strncpy_s_l', '_snprintf_s', 'CC_SHA512_Init', 'fwscanf_s', '_snwprintf_s', 'CC_SHA1', 'swprintf', 
           'fprintf', 'EVP_DigestInit_ex', 'strlen', 'SHA1_Init', 'strncat', '_getws_s', 'CC_MD4_Final', 
           'wnsprintfW', 'lcong48', 'lrand48', 'write', 'HMAC_Init', '_wfopen_s', 'wmemchr', '_tmakepath', 
           'wnsprintfA', 'lstrcpynW', 'scanf_s', '_mbsncpy_s_l', '_localtime64_s', 'fstream.open', '_wmakepath', 
           'Connection.open', '_tccat', 'valloc', 'setgroups', 'unlink', 'fstream.put', 'wsprintfA', '*SHA1*', 
           '_wsearchenv_s', 'ualstrcpyA', 'CC_MD5_Update', 'strerror_s', 'HeapCreate', 'ualstrcpyW', '__xstat', 
           '_wmktemp_s', 'StrCatChainW', 'ldap_search_st', '_mbstowcs_s_l', 'ldap_modify_ext', '_mbsset_s', 
           'strncpy_s', 'move', 'execle', 'StrCat', 'xrealloc', 'wcsncpy_s', '_tcsncpy*', 'execlp', 
           'RIPEMD160_Final', 'ldap_search_s', 'EnterCriticalSection', '_wctomb_s_l', 'fwrite', '_gmtime64_s', 
           'sscanf_s', 'wcscat', '_strupr_s', 'wcrtomb_s', 'VirtualLock', 'ldap_add_ext_s', '_mbscpy', 
           '_localtime32_s', 'lstrcpy', '_wcsncpy*', 'CC_SHA1_Init', '_getts', '_wfopen', '__xstat64', 
           'strcoll', '_fwscanf_s_l', '_mbslwr_s_l', 'RegOpenKey', 'makepath', 'seed48', 'CC_SHA256', 
           'sendto', 'execv', 'CalculateDigest', 'memchr', '_mbscpy_s', '_strtime_s', 'ldap_search_ext_s', 
           '_chmod', 'flock', '__fxstat64', '_vsntprintf', 'CC_SHA256_Init', '_itoa_s', '__wcserror_s', 
           '_gcvt_s', 'fstream.write', 'sprintf', 'recursive_mutex', 'strrchr', 'gethostbyaddr', '_wcsupr_s_l', 
           'strcspn', 'MD5_Final', 'asprintf', '_wcstombs_s_l', '_tcstok', 'free', 'MD2_Final', 'asctime_s', 
           '_alloca', '_wputenv_s', '_wcsset_s', '_wcslwr_s_l', 'SHA1_Update', 'filebuf.sputc', 'filebuf.sputn', 
           'SQLConnect', 'ldap_compare', 'mbstowcs_s', 'HMAC_Final', 'pthread_condattr_init', '_ultow_s', 'rand', 
           'ofstream.put', 'CC_SHA224_Final', 'lstrcpynA', 'bcopy', 'system', 'CreateFile*', 'wcscpy_s', 
           '_mbsnbcpy*', 'open', '_vsnwprintf', 'strncpy', 'getopt_long', 'CC_SHA512_Final', '_vsprintf_s_l', 
           'scanf', 'mkdir', '_localtime_s', '_snprintf', '_mbccpy_s_l', 'memcmp', 'final', '_ultoa_s', 
           'lstrcpyW', 'LoadModule', '_swprintf_s_l', 'MD5_Update', '_mbsnset_s_l', '_wstrtime_s', '_strnset_s', 
           'lstrcpyA', '_mbsnbcpy_s', 'mlock', 'IsBadHugeWritePtr', 'copy', '_mbsnbcpy_s_l', 'wnsprintf', 
           'wcscpy', 'ShellExecute', 'CC_MD4', '_ultow', '_vsnwprintf_s_l', 'lstrcpyn', 'CC_SHA1_Final', 
           'vsnprintf', '_mbsnbset_s', '_i64tow', 'SHA256_Init', 'wvnsprintf', 'RegCreateKey', 'strtok_s', 
           '_wctime32_s', '_i64toa', 'CC_MD5_Final', 'wmemcpy', 'WinExec', 'CreateDirectory*', 
           'CC_SHA256_Update', '_vsnprintf_s_l', 'jrand48', 'wsprintf', 'ldap_rename_ext_s', 'filebuf.open', 
           '_wsystem', 'SHA256_Update', '_cwscanf_s', 'wsprintfW', '_sntscanf', '_splitpath', 'fscanf_s', 
           'strpbrk', 'wcstombs_s', 'wscanf', '_mbsnbcat_s_l', 'strcpynA', 'pthread_cond_init', 'wcsrtombs_s', 
           '_wsopen_s', 'CharToOemBuffA', 'RIPEMD160_Update', '_tscanf', 'HMAC', 'StrCCpy', 'Connection.connect', 
           'lstrcatn', '_mbstok', '_mbsncpy', 'CC_SHA384_Update', 'create_directories', 'pthread_mutex_unlock', 
           'CFile.Open', 'connect', '_vswprintf_s_l', '_snscanf_s_l', 'fputc', '_wscanf_s', '_snprintf_s_l', 
           'strtok', '_strtok_s_l', 'lstrcatA', 'snwscanf', 'pthread_mutex_init', 'fputs', 'CC_SHA384_Init', 
           '_putenv_s', 'CharToOemBuffW', 'pthread_mutex_trylock', '__wcstoul_internal', '_memccpy', 
           '_snwprintf_s_l', '_strncpy*', 'wmemset', 'MD4_Init', '*RC4*', 'strcpyW', '_ecvt_s', 'memcpy_s', 
           'erand48', 'IsBadHugeReadPtr', 'strcpyA', 'HeapReAlloc', 'memcpy', 'ldap_rename_ext', 'fopen_s', 
           'srandom', '_cgetws_s', '_makepath', 'SHA256_Final', 'remove', '_mbsupr_s', 'pthread_mutexattr_init', 
           '__wcstold_internal', 'StrCpy', 'ldap_delete', 'wmemmove_s', '_mkdir', 'strcat', '_cscanf_s_l', 
           'StrCAdd', 'swprintf_s', '_strnset_s_l', 'close', 'ldap_delete_ext_s', 'ldap_modrdn', 'strchr', 
           '_gmtime32_s', '_ftcscat', 'lstrcatnA', '_tcsncat', 'OemToChar', 'mutex', 'CharToOem', 'strcpy_s', 
           'lstrcatnW', '_wscanf_s_l', '__lxstat64', 'memalign', 'MD2_Init', 'StrCatBuffW', 'StrCpyN', 'CC_MD5', 
           'StrCpyA', 'StrCatBuffA', 'StrCpyW', 'tmpnam_r', '_vsnprintf', 'strcatA', 'StrCpyNW', '_mbsnbset_s_l', 
           'EVP_DigestInit', '_stscanf', 'CC_MD2', '_tcscat', 'StrCpyNA', 'xmalloc', '_tcslen', '*MD4*', 
           'vasprintf', 'strxfrm', 'chmod', 'ldap_add_ext', 'alloca', '_snscanf_s', 'IsBadWritePtr', 'swscanf_s', 
           'wmemcpy_s', '_itoa', '_ui64toa_s', 'EVP_DigestUpdate', '__wcstol_internal', '_itow', 'StrNCatW', 
           'strncat_s', 'ualstrcpy', 'execvp', '_mbccat', 'EVP_MD_CTX_init', 'assert', 'ofstream.write', 
           'ldap_add', '_sscanf_s_l', 'drand48', 'CharToOemW', 'swscanf', '_itow_s', 'RIPEMD160_Init', 
           'CopyMemory', 'initstate', 'getpwuid', 'vsprintf', '_fcvt_s', 'CharToOemA', 'setuid', 'malloc', 
           'StrCatNA', 'strcat_s', 'srand', 'getwd', '_controlfp_s', 'olestrcpy', '__wcstod_internal', 
           '_mbsnbcat', 'lstrncat', 'des_*', 'CC_SHA224_Init', 'set*', 'vsprintf_s', 'SHA1_Final', '_umask_s', 
           'gets', 'setstate', 'wvsprintfW', 'LoadLibraryEx', 'ofstream.open', 'calloc', '_mbstrlen', 
           '_cgets_s', '_sopen_s', 'IsBadStringPtr', 'wcsncat_s', 'add*', 'nrand48', 'create_directory', 
           'ldap_search_ext', '_i64toa_s', '_ltoa_s', '_cwscanf_s_l', 'wmemcmp', '__lxstat', 'lstrlen', 
           'pthread_condattr_destroy', '_ftcscpy', 'wcstok_s', '__xmknod', 'pthread_attr_destroy', 'sethostname', 
           '_fscanf_s_l', 'StrCatN', 'RegEnumKey', '_tcsncpy', 'strcatW', 'AfxLoadLibrary', 'setenv', 'tmpnam', 
           '_mbsncat_s_l', '_wstrdate_s', '_wctime64_s', '_i64tow_s', 'CC_MD4_Update', 'ldap_add_s', '_umask', 
           'CC_SHA1_Update', '_wcsset_s_l', '_mbsupr_s_l', 'strstr', '_tsplitpath', 'memmove', '_tcscpy', 
           'vsnprintf_s', 'strcmp', 'wvnsprintfW', 'tmpfile', 'ldap_modify', '_mbsncat*', 'mrand48', 'sizeof', 
           'StrCatA', '_ltow_s', '*desencrypt*', 'StrCatW', '_mbccpy', 'CC_MD2_Init', 'RIPEMD160', 'ldap_search', 
           'CC_SHA224', 'mbsrtowcs_s', 'update', 'ldap_delete_s', 'getnameinfo', '*RC5*', '_wcsncat_s_l', 
           'DriverManager.getConnection', 'socket', '_cscanf_s', 'ldap_modrdn_s', '_wopen', 'CC_SHA256_Final', 
           '_snwprintf*', 'MD2_Update', 'strcpy', '_strncat_s_l', 'CC_MD5_Init', 'mbscpy', 'wmemmove', 
           'LoadLibraryW', '_mbslen', '*alloc', '_mbsncat_s', 'LoadLibraryA', 'fopen', 'StrLen', 'delete', 
           '_splitpath_s', 'CreateFileTransacted*', 'MD4_Final', '_open', 'CC_SHA384', 'wcslen', 'wcsncat', 
           '_mktemp_s', 'pthread_mutexattr_destroy', '_snwscanf_s', '_strset_s', '_wcsncpy_s_l', 'CC_MD2_Final', 
           '_mbstok_s_l', 'wctomb_s', 'MySQL_Driver.connect', '_snwscanf_s_l', '*_des_*', 'LoadLibrary', 
           '_swscanf_s_l', 'ldap_compare_s', 'ldap_compare_ext', '_strlwr_s', 'GetEnvironmentVariable', 
           'cuserid', '_mbscat_s', 'strspn', '_mbsncpy_s', 'ldap_modrdn2', 'LeaveCriticalSection', 'CopyFile', 
           'getpwd', 'sscanf', 'creat', 'RegSetValue', 'ldap_modrdn2_s', 'CFile.Close', '*SHA_1*', 
           'pthread_cond_destroy', 'CC_SHA512_Update', '*RC2*', 'StrNCatA', '_mbsnbcpy', '_mbsnset_s', 
           'crypt', 'excel', '_vstprintf', 'xstrdup', 'wvsprintfA', 'getopt', 'mkstemp', '_wcsnset_s', 
           '_stprintf', '_sntprintf', 'tmpfile_s', 'OpenDocumentFile', '_mbsset_s_l', '_strset_s_l', 
           '_strlwr_s_l', 'ifstream.open', 'xcalloc', 'StrNCpyA', '_wctime_s', 'CC_SHA224_Update', '_ctime64_s', 
           'MoveFile', 'chown', 'StrNCpyW', 'IsBadReadPtr', '_ui64tow_s', 'IsBadCodePtr', 'getc', 
           'OracleCommand.ExecuteOracleScalar', 'AccessDataSource.Insert', 'IDbDataAdapter.FillSchema', 
           'IDbDataAdapter.Update', 'GetWindowText*', 'SendMessage', 'SqlCommand.ExecuteNonQuery', 'streambuf.sgetc', 
           'streambuf.sgetn', 'OracleCommand.ExecuteScalar', 'SqlDataSource.Update', '_Read_s', 'IDataAdapter.Fill', 
           '_wgetenv', '_RecordsetPtr.Open*', 'AccessDataSource.Delete', 'Recordset.Open*', 'filebuf.sbumpc', 'DDX_*', 
           'RegGetValue', 'fstream.read*', 'SqlCeCommand.ExecuteResultSet', 'SqlCommand.ExecuteXmlReader', 'main', 
           'streambuf.sputbackc', 'read', 'm_lpCmdLine', 'CRichEditCtrl.Get*', 'istream.putback', 
           'SqlCeCommand.ExecuteXmlReader', 'SqlCeCommand.BeginExecuteXmlReader', 'filebuf.sgetn', 
           'OdbcDataAdapter.Update', 'filebuf.sgetc', 'SQLPutData', 'recvfrom', 'OleDbDataAdapter.FillSchema', 
           'IDataAdapter.FillSchema', 'CRichEditCtrl.GetLine', 'DbDataAdapter.Update', 'SqlCommand.ExecuteReader', 
           'istream.get', 'ReceiveFrom', '_main', 'fgetc', 'DbDataAdapter.FillSchema', 'kbhit', 'UpdateCommand.Execute*', 
           'Statement.execute', 'fgets', 'SelectCommand.Execute*', 'getch', 'OdbcCommand.ExecuteNonQuery', 
           'CDaoQueryDef.Execute', 'fstream.getline', 'ifstream.getline', 'SqlDataAdapter.FillSchema', 
           'OleDbCommand.ExecuteReader', 'Statement.execute*', 'SqlCeCommand.BeginExecuteNonQuery', 
           'OdbcCommand.ExecuteScalar', 'SqlCeDataAdapter.Update', 'sendmessage', 'mysqlpp.DBDriver', 'fstream.peek', 
           'Receive', 'CDaoRecordset.Open', 'OdbcDataAdapter.FillSchema', '_wgetenv_s', 'OleDbDataAdapter.Update', 
           'readsome', 'SqlCommand.BeginExecuteXmlReader', 'recv', 'ifstream.peek', '_Main', '_tmain', '_Readsome_s', 
           'SqlCeCommand.ExecuteReader', 'OleDbCommand.ExecuteNonQuery', 'fstream.get', 'IDbCommand.ExecuteScalar', 
           'filebuf.sputbackc', 'IDataAdapter.Update', 'streambuf.sbumpc', 'InsertCommand.Execute*', 'RegQueryValue', 
           'IDbCommand.ExecuteReader', 'SqlPipe.ExecuteAndSend', 'Connection.Execute*', 'getdlgtext', 'ReceiveFromEx', 
           'SqlDataAdapter.Update', 'RegQueryValueEx', 'SQLExecute', 'pread', 'SqlCommand.BeginExecuteReader', 'AfxWinMain', 
           'getchar', 'istream.getline', 'SqlCeDataAdapter.Fill', 'OleDbDataReader.ExecuteReader', 'SqlDataSource.Insert', 
           'istream.peek', 'SendMessageCallback', 'ifstream.read*', 'SqlDataSource.Select', 'SqlCommand.ExecuteScalar', 
           'SqlDataAdapter.Fill', 'SqlCommand.BeginExecuteNonQuery', 'getche', 'SqlCeCommand.BeginExecuteReader', 'getenv', 
           'streambuf.snextc', 'Command.Execute*', '_CommandPtr.Execute*', 'SendNotifyMessage', 'OdbcDataAdapter.Fill', 
           'AccessDataSource.Update', 'fscanf', 'QSqlQuery.execBatch', 'DbDataAdapter.Fill', 'cin', 
           'DeleteCommand.Execute*', 'QSqlQuery.exec', 'PostMessage', 'ifstream.get', 'filebuf.snextc', 
           'IDbCommand.ExecuteNonQuery', 'Winmain', 'fread', 'getpass', 'GetDlgItemTextCCheckListBox.GetCheck', 
           'DISP_PROPERTY_EX', 'pread64', 'Socket.Receive*', 'SACommand.Execute*', 'SQLExecDirect', 
           'SqlCeDataAdapter.FillSchema', 'DISP_FUNCTION', 'OracleCommand.ExecuteNonQuery', 'CEdit.GetLine', 
           'OdbcCommand.ExecuteReader', 'CEdit.Get*', 'AccessDataSource.Select', 'OracleCommand.ExecuteReader', 
           'OCIStmtExecute', 'getenv_s', 'DB2Command.Execute*', 'OracleDataAdapter.FillSchema', 'OracleDataAdapter.Fill', 
           'CComboBox.Get*', 'SqlCeCommand.ExecuteNonQuery', 'OracleCommand.ExecuteOracleNonQuery', 'mysqlpp.Query', 
           'istream.read*', 'CListBox.GetText', 'SqlCeCommand.ExecuteScalar', 'ifstream.putback', 'readlink', 
           'CHtmlEditCtrl.GetDHtmlDocument', 'PostThreadMessage', 'CListCtrl.GetItemText', 'OracleDataAdapter.Update', 
           'OleDbCommand.ExecuteScalar', 'stdin', 'SqlDataSource.Delete', 'OleDbDataAdapter.Fill', 'fstream.putback', 
           'IDbDataAdapter.Fill', '_wspawnl', 'fwprintf', 'sem_wait', '_unlink', 'ldap_search_ext_sW', 'signal', 'PQclear', 
           'PQfinish', 'PQexec', 'PQresultStatus']
keywords = ["alignas", "alignof", "and", "and_eq", "asm", "atomic_cancel", "atomic_commit", 
            "atomic_noexcept", "auto", "bitand", "bitor", "bool", "break", "case", "catch", 
            "char", "char8_t", "char16_t", "char32_t", "class", "compl", "concept", "const", 
            "consteval", "constexpr", "constinit", "const_cast", "continue", "co_await", 
            "co_return", "co_yield", "decltype", "default", "delete", "do", "double", "dynamic_cast", 
            "else", "enum", "explicit", "export", "extern", "false", "float", "for", "friend", "goto", 
            "if", "inline", "int", "long", "mutable", "namespace", "new", "noexcept", "not", "not_eq", 
            "nullptr", "operator", "or", "or_eq", "private", "protected", "public", "reflexpr", 
            "register", "reinterpret_cast", "requires", "return", "short", "signed", "sizeof", "static", 
            "static_assert", "static_cast", "struct", "switch", "synchronized", "template", "this", 
            "thread_local", "throw", "true", "try", "typedef", "typeid", "typename", "union", "unsigned", 
            "using", "virtual", "void", "volatile", "wchar_t", "while", "xor", "xor_eq", "NULL"]
puncs = list('~`!@#$%^&*()-+={[]}|\\;:\'\"<,>.?/')

def read_csv(csv_file_path):
    data = []
    with open(csv_file_path) as fp:
        header = fp.readline()
        header = header.strip()
        h_parts = [hp.strip() for hp in header.split('\t')]
        for line in fp:
            line = line.strip()
            instance = {}
            lparts = line.split('\t')
            for i, hp in enumerate(h_parts):
                if i < len(lparts):
                    content = lparts[i].strip()
                else:
                    content = ''
                instance[hp] = content
            data.append(instance)
        return data


def read_code_file(file_path):
    code_lines = {}
    with open(file_path) as fp:
        for ln, line in enumerate(fp):
            assert isinstance(line, str)
            line = line.strip()
            if '//' in line:
                line = line[:line.index('//')]
            code_lines[ln + 1] = line
        return code_lines


def extract_nodes_with_location_info(nodes):
    # Will return an array identifying the indices of those nodes in nodes array,
    # another array identifying the node_id of those nodes
    # another array indicating the line numbers
    # all 3 return arrays should have same length indicating 1-to-1 matching.
    node_indices = []
    node_ids = []
    line_numbers = []
    node_id_to_line_number = {}
    for node_index, node in enumerate(nodes):
        assert isinstance(node, dict)
        if 'location' in node.keys():
            location = node['location']
            if location == '':
                continue
            line_num = int(location.split(':')[0])
            node_id = node['key'].strip()
            node_indices.append(node_index)
            node_ids.append(node_id)
            line_numbers.append(line_num)
            node_id_to_line_number[node_id] = line_num
    return node_indices, node_ids, line_numbers, node_id_to_line_number
    pass


def create_adjacency_list(line_numbers, node_id_to_line_numbers, edges, data_dependency_only=False):
    adjacency_list = {}
    for ln in set(line_numbers):
        adjacency_list[ln] = [set(), set()]
    for edge in edges:
        edge_type = edge['type'].strip()
        if True :#edge_type in ['IS_AST_PARENT', 'FLOWS_TO']:
            start_node_id = edge['start'].strip()
            end_node_id = edge['end'].strip()
            if start_node_id not in node_id_to_line_numbers.keys() or end_node_id not in node_id_to_line_numbers.keys():
                continue
            start_ln = node_id_to_line_numbers[start_node_id]
            end_ln = node_id_to_line_numbers[end_node_id]
            if not data_dependency_only:
                if edge_type == 'CONTROLS': #Control Flow edges
                    adjacency_list[start_ln][0].add(end_ln)
            if edge_type == 'REACHES': # Data Flow edges
                adjacency_list[start_ln][1].add(end_ln)
    return adjacency_list


def create_visual_graph(code, adjacency_list, file_name='test_graph', verbose=False):
    graph = Digraph('Code Property Graph')
    for ln in adjacency_list:
        graph.node(str(ln), str(ln) + '\t' + code[ln], shape='box')
        control_dependency, data_dependency = adjacency_list[ln]
        for anode in control_dependency:
            graph.edge(str(ln), str(anode), color='red')
        for anode in data_dependency:
            graph.edge(str(ln), str(anode), color='blue')
    graph.render(file_name, view=verbose)


def create_forward_slice(adjacency_list, line_no):
    sliced_lines = set()
    sliced_lines.add(line_no)
    stack = list()
    stack.append(line_no)
    while len(stack) != 0:
        cur = stack.pop()
        if cur not in sliced_lines:
            sliced_lines.add(cur)
        adjacents = adjacency_list[cur]
        for node in adjacents:
            if node not in sliced_lines:
                stack.append(node)
    sliced_lines = sorted(sliced_lines)
    return sliced_lines


def combine_control_and_data_adjacents(adjacency_list):
    cgraph = {}
    for ln in adjacency_list:
        cgraph[ln] = set()
        cgraph[ln] = cgraph[ln].union(adjacency_list[ln][0])
        cgraph[ln] = cgraph[ln].union(adjacency_list[ln][1])
    return cgraph


def invert_graph(adjacency_list):
    igraph = {}
    for ln in adjacency_list.keys():
        igraph[ln] = set()
    for ln in adjacency_list:
        adj = adjacency_list[ln]
        for node in adj:
            igraph[node].add(ln)
    return igraph
    pass


def create_backward_slice(adjacency_list, line_no):
    inverted_adjacency_list = invert_graph(adjacency_list)
    return create_forward_slice(inverted_adjacency_list, line_no)


class Tokenizer:
    # creates the object, does the inital parse
    def __init__(self, path, tokenizer_type='original'):
        self.index = clang.cindex.Index.create()
        self.tu = self.index.parse(path)
        self.path = self.extract_path(path)
        self.symbol_table = {}
        self.symbol_count = 1
        self.tokenizer_type = tokenizer_type

    # To output for split_functions, must have same path up to last two folders
    def extract_path(self, path):
        return "".join(path.split("/")[:-2])

    
    def full_tokenize_cursor(self, cursor):
        tokens = cursor.get_tokens()
        result = []
        for token in tokens:
            if token.kind.name == "COMMENT":
                continue
            if token.kind.name == "LITERAL":
                result += self.process_literal(token)
                continue
            if token.kind.name == "IDENTIFIER":
                result += ["ID"]
                continue
            result += [token.spelling]
        return result

    def full_tokenize(self):
        cursor = self.tu.cursor
        return self.full_tokenize_cursor(cursor)

    def process_literal(self, literal):
        cursor_kind = clang.cindex.CursorKind
        kind = literal.cursor.kind
        if kind == cursor_kind.INTEGER_LITERAL:
            return literal.spelling
        if kind == cursor_kind.FLOATING_LITERAL:
            return literal.spelling
        if kind == cursor_kind.IMAGINARY_LITERAL:
            return ["NUM"]       
        if kind == cursor_kind.STRING_LITERAL:
            return ["STRING"]
        sp = literal.spelling
        if re.match('[0-9]+', sp) is not None:
            return sp
        return ["LITERAL"]

    def split_functions(self, method_only):
        results = []
        cursor_kind = clang.cindex.CursorKind
        cursor = self.tu.cursor
        for c in cursor.get_children():
            filename = c.location.file.name if c.location.file != None else "NONE"
            extracted_path = self.extract_path(filename)
            # print(c.kind)
            if (c.kind == cursor_kind.CXX_METHOD or (method_only == False and c.kind == cursor_kind.FUNCTION_DECL)) and extracted_path == self.path:
                name = c.spelling
                tokens = self.full_tokenize_cursor(c)
                filename = filename.split("/")[-1]
                results += [tokens]

        return results
    

def tokenize(file_text):
    # Kiểm tra nếu thư mục ./tmp không tồn tại, thì tạo mới
    if not os.path.exists('./tmp'):
        os.makedirs('./tmp')
    
    # Tiến hành ghi tệp sau khi đảm bảo thư mục tồn tại
    c_file = open('./tmp/test.c', 'w+')
    c_file.write(file_text)
    c_file.close()
    
    tok = Tokenizer('./tmp/test.c')
    results = tok.split_functions(False)
    if results == []:
        return None
    return ' '.join(results[0])

def read_file(path):
    with open(path) as f:
        lines = f.readlines()
        return ' '.join(lines)
    
def extract_line_number(idx, nodes):
    while idx >= 0:
        c_node = nodes[idx]
        if 'location' in c_node.keys():
            location = c_node['location']
            if location.strip() != '':
                try:
                    ln = int(location.split(':')[0])
                    return ln
                except:
                    pass
        idx -= 1
    return -1

def symbolic_tokenize(code):
    tokens = nltk.word_tokenize(code)
    c_tokens = []
    for t in tokens:
        if t.strip() != '':
            c_tokens.append(t.strip())
    f_count = 1
    var_count = 1
    symbol_table = {}
    final_tokens = []
    for idx in range(len(c_tokens)):
        t = c_tokens[idx]
        if t in keywords:
            final_tokens.append(t)
        elif t in puncs:
            final_tokens.append(t)
        elif t in l_funcs:
            final_tokens.append(t)
        elif idx < len(c_tokens) - 1 and c_tokens[idx + 1] == '(':
            if t in keywords:
                final_tokens.append(t)
            else:
                if t not in symbol_table.keys():
                    symbol_table[t] = "FUNC" + str(f_count)
                    f_count += 1
                final_tokens.append(symbol_table[t])
            idx += 1
        
        elif t.endswith('('):
            t = t[:-1]
            if t in keywords:
                final_tokens.append(t + '(')
            else:
                if t not in symbol_table.keys():
                    symbol_table[t] = "FUNC" + str(f_count)
                    f_count += 1
                final_tokens.append(symbol_table[t] + '(')
        elif t.endswith('()'):
            t = t[:-2]
            if t in keywords:
                final_tokens.append(t + '()')
            else:
                if t not in symbol_table.keys():
                    symbol_table[t] = "FUNC" + str(f_count)
                    f_count += 1
                final_tokens.append(symbol_table[t] + '()')
        elif re.match("^\"*\"$", t) is not None:
            final_tokens.append("STRING")
        elif re.match("^[0-9]+(\.[0-9]+)?$", t) is not None:
            final_tokens.append("NUMBER")
        elif re.match("^[0-9]*(\.[0-9]+)$", t) is not None:
            final_tokens.append("NUMBER")
        else:
            if t not in symbol_table.keys():
                symbol_table[t] = "VAR" + str(var_count)
                var_count += 1
            final_tokens.append(symbol_table[t])
    return ' '.join(final_tokens)

def reformat_code_line_graph(code_lines, adjacency_lists, lanel, wv_model_original, wv_model_li, label):
    actual_lines = []
    for ln in adjacency_lists.keys():
        cd, dd = adjacency_lists[ln]
        new_cd = [l for l in cd]
        new_dd = [l for l in dd] 
        actual_lines.extend(new_cd)
        actual_lines.extend(new_dd)
        actual_lines.append(ln)
    actual_lines = sorted(list(set(actual_lines)))
    line_no_to_idx = {}
    idx_to_line_no = {}
    for idx, ln in enumerate(actual_lines):
        line_no_to_idx[ln] = idx
        idx_to_line_no[idx] = ln
    data_point = {}
    graph = []
    for src in adjacency_lists.keys():
        cd, dd = adjacency_lists[src]
        for dest in cd:
            graph.append([line_no_to_idx[src], 0, line_no_to_idx[dest]])
            graph.append([line_no_to_idx[dest], 1, line_no_to_idx[src]])
        for dest in dd:
            graph.append([line_no_to_idx[src], 2, line_no_to_idx[dest]])
            graph.append([line_no_to_idx[dest], 3, line_no_to_idx[src]])
    original_tokens = []
    symbolic_tokens = []
    line_features_wv = []
    sym_line_features_wv = []
    
    for lidx in range(len(idx_to_line_no.keys())):
        actual_code_line = code_lines[idx_to_line_no[lidx]]
        actual_line_tokens = nltk.wordpunct_tokenize(actual_code_line)
        symbolic_line_tokens = symbolic_tokenize(actual_code_line).split()
        original_tokens.append(actual_line_tokens)
        symbolic_tokens.append(symbolic_line_tokens)
        
        nrp = np.zeros(100)
        for token in actual_line_tokens:
            try:
                embedding = wv_model_original.wv[token]
            except:
                embedding = np.zeros(100)
            nrp = np.add(nrp, embedding)
        if len(actual_line_tokens) > 0:
            fNrp = np.divide(nrp, len(actual_line_tokens))
        else:
            fNrp = nrp
        line_features_wv.append(fNrp.tolist())
        
        nrp = np.zeros(64)
        for token in symbolic_line_tokens:
            try:
                embedding = wv_model_li.wv[token]
            except:
                embedding = np.zeros(64)
            nrp = np.add(nrp, embedding)
        if len(actual_line_tokens) > 0:
            fNrp = np.divide(nrp, len(symbolic_line_tokens))
        else:
            fNrp = nrp
        sym_line_features_wv.append(fNrp.tolist())
    data_point = {
        'node_features': line_features_wv,
        'node_features_sym': sym_line_features_wv,
        'graph': graph,
        'original_tokens': original_tokens,
        'symbolic_tokens': symbolic_tokens,
        'targets': [[label]]
    }
    return data_point

def extract_slices(project):
    # Đường dẫn đến thư mục chứa mã thô và thư mục chứa mã đã được phân tích
    split_dir = f'/home/p4ultr4n/workplace/ReVeal/raw_code/'
    parsed = f'/home/p4ultr4n/workplace/ReVeal/code-slicer/parsed/raw_code/'
    all_data = []  # Danh sách để lưu tất cả dữ liệu sau khi xử lý

    # Đọc file JSON chứa tên file mã nguồn để xử lý
    ggnn_json_data = json.load(open(f'/home/p4ultr4n/workplace/ReVeal/data/devign.json'))
    files = [d['file_name'] for d in ggnn_json_data]  # Lấy danh sách các file mã nguồn từ dữ liệu JSON
    print(len(files))  # In ra số lượng file

    # Lặp qua từng file mã nguồn để phân tích slice
    for i, file_name in enumerate(files):
        print(file_name.strip())
        # Lấy nhãn của file từ tên file (0: lành tính, 1: lỗ hổng)
        label = file_name.strip()[:-2].split('_')[-1]
        code_text = read_file(split_dir + file_name.strip())  # Đọc nội dung của file mã nguồn
        
        # Đường dẫn tới các file nodes và edges đã phân tích
        nodes_file_path = parsed + file_name.strip() + '/nodes.csv'
        edges_file_path = parsed + file_name.strip() + '/edges.csv'
        
        # Đọc các nodes từ file CSV và tạo các tập hợp lưu trữ dòng lệnh theo loại
        nc = open(nodes_file_path)
        nodes_file = csv.DictReader(nc, delimiter='\t')
        nodes = [node for node in nodes_file]  # Lưu tất cả nodes vào danh sách
        call_lines, array_lines, ptr_lines, arithmatic_lines = set(), set(), set(), set()
        
        # Nếu không có node nào, bỏ qua file này
        if len(nodes) == 0:
            continue

        # Xác định các dòng lệnh gọi hàm, truy cập mảng, con trỏ, và phép toán số học từ nodes
        for node_idx, node in enumerate(nodes):
            ntype = node['type'].strip()
            if ntype == 'CallExpression':  # Kiểm tra loại node có phải là CallExpression không
                function_name = nodes[node_idx + 1]['code']
                if function_name is None or function_name.strip() == '':
                    continue
                if function_name.strip() in l_funcs:  # Nếu là hàm cần tìm, thêm dòng vào tập hợp call_lines
                    line_no = extract_line_number(node_idx, nodes)
                    if line_no > 0:
                        call_lines.add(line_no)
            elif ntype == 'ArrayIndexing':  # Kiểm tra nếu node là ArrayIndexing
                line_no = extract_line_number(node_idx, nodes)
                if line_no > 0:
                    array_lines.add(line_no)
            elif ntype == 'PtrMemberAccess':  # Kiểm tra nếu node là PtrMemberAccess
                line_no = extract_line_number(node_idx, nodes)
                if line_no > 0:
                    ptr_lines.add(line_no)
            elif node['operator'].strip() in ['+', '-', '*', '/']:  # Kiểm tra toán tử số học
                line_no = extract_line_number(node_idx, nodes)
                if line_no > 0:
                    arithmatic_lines.add(line_no)

        # Đọc lại các nodes và edges để tạo đồ thị biểu diễn slice
        nodes = read_csv(nodes_file_path)
        edges = read_csv(edges_file_path)
        node_indices, node_ids, line_numbers, node_id_to_ln = extract_nodes_with_location_info(nodes)
        adjacency_list = create_adjacency_list(line_numbers, node_id_to_ln, edges, False)  # Tạo danh sách kề
        combined_graph = combine_control_and_data_adjacents(adjacency_list)  # Kết hợp điều khiển và dữ liệu đồ thị

        # Biến lưu trữ slice cho từng loại dòng lệnh
        array_slices, array_slices_bdir = [], []
        call_slices, call_slices_bdir = [], []
        arith_slices, arith_slices_bdir = [], []
        ptr_slices, ptr_slices_bdir = [], []
        all_slices = []
        all_keys, _keys = set(), set()

        # Tạo slice cho các dòng lệnh gọi hàm
        for slice_ln in call_lines:
            forward_sliced_lines = create_forward_slice(combined_graph, slice_ln)
            backward_sliced_lines = create_backward_slice(combined_graph, slice_ln)
            all_slice_lines = sorted(list(set(forward_sliced_lines + backward_sliced_lines)))
            key = ' '.join([str(i) for i in all_slice_lines])
            if key not in _keys:
                call_slices.append(backward_sliced_lines)  # Thêm slice theo hướng ngược
                call_slices_bdir.append(all_slice_lines)  # Thêm slice theo hướng bất kỳ
                _keys.add(key)
            if key not in all_keys:
                all_slices.append(all_slice_lines)
                all_keys.add(key)

        # Tạo slice cho các dòng lệnh truy cập mảng
        _keys = set()
        for slice_ln in array_lines:
            forward_sliced_lines = create_forward_slice(combined_graph, slice_ln)
            backward_sliced_lines = create_backward_slice(combined_graph, slice_ln)
            all_slice_lines = sorted(list(set(forward_sliced_lines + backward_sliced_lines)))
            key = ' '.join([str(i) for i in all_slice_lines])
            if key not in _keys:
                array_slices.append(backward_sliced_lines)
                array_slices_bdir.append(all_slice_lines)
                _keys.add(key)
            if key not in all_keys:
                all_slices.append(all_slice_lines)
                all_keys.add(key)

        # Tạo slice cho các dòng lệnh số học
        _keys = set()
        for slice_ln in arithmatic_lines:
            forward_sliced_lines = create_forward_slice(combined_graph, slice_ln)
            backward_sliced_lines = create_backward_slice(combined_graph, slice_ln)
            all_slice_lines = sorted(list(set(forward_sliced_lines + backward_sliced_lines)))
            key = ' '.join([str(i) for i in all_slice_lines])
            if key not in _keys:
                arith_slices.append(backward_sliced_lines)
                arith_slices_bdir.append(all_slice_lines)
                _keys.add(key)
            if key not in all_keys:
                all_slices.append(all_slice_lines)
                all_keys.add(key)

        # Tạo slice cho các dòng lệnh truy cập con trỏ
        _keys = set()
        for slice_ln in ptr_lines:
            forward_sliced_lines = create_forward_slice(combined_graph, slice_ln)
            backward_sliced_lines = create_backward_slice(combined_graph, slice_ln)
            all_slice_lines = sorted(list(set(forward_sliced_lines + backward_sliced_lines)))
            key = ' '.join([str(i) for i in all_slice_lines])
            if key not in _keys:
                ptr_slices.append(backward_sliced_lines)
                ptr_slices_bdir.append(all_slice_lines)
                _keys.add(key)
            if key not in all_keys:
                all_slices.append(all_slice_lines)
                all_keys.add(key)

        # Tokenize mã nguồn và thêm dữ liệu vào all_data
        print("Debug1")
        t_code = tokenize(code_text)  # Chuyển mã thành token
        if t_code is None:
            continue
        data_instance = {
            'file_path': split_dir + file_name.strip(),
            'code': code_text,
            'tokenized': t_code,
            'call_slices_vd': call_slices,
            'call_slices_sy': call_slices_bdir,
            'array_slices_vd': array_slices,
            'array_slices_sy': array_slices_bdir,
            'arith_slices_vd': arith_slices,
            'arith_slices_sy': arith_slices_bdir,
            'ptr_slices_vd': ptr_slices,
            'ptr_slices_sy': ptr_slices_bdir,
            'label': int(label)  # Gán nhãn cho dữ liệu
        }
        all_data.append(data_instance)

    # Lưu toàn bộ dữ liệu vào file JSON
    print(len(all_data))  # In ra số lượng phần tử trong dữ liệu
    output_file = open(f'/home/p4ultr4n/workplace/ReVeal/data/{project}_full_data_with_slices.json', 'w+')
    print("Success")
    json.dump(all_data, output_file)
    output_file.close()
    return all_data

def extract_graph_data(portion, 
    project, base_dir='/home/ding/dlvp/dl-vulnerability-detection/data/commits/code/output/', 
    output_dir='/home/ding/dlvp/dl-vulnerability-detection/data/commits/code/full_experiment_real_data_processed/'):
    assert portion in ['full_graph', 'cgraph', 'dgraph', 'cdgraph']
    shards = os.listdir(os.path.join(base_dir, project))
    shard_count = len(shards)
    total_functions, in_scope_function = set(), set()
    vnt, nvnt = 0, 0
    graphs = []
    for sc in range(1, shard_count + 1):
        shard_file = open(os.path.join(base_dir, project, project + '.json.shard' + str(sc)))
        shard_data = json.load(shard_file)
        for data in tqdm(shard_data):
            fidx = data['id']
            label = int(data['label'])
            total_functions.add(fidx)
            present = data[portion] is not None
            code_graph = data[portion]
            if present:
                code_graph['id'] = fidx
                code_graph['file_name'] = data['file_name']
                code_graph['file_path'] = data['file_path']
                code_graph['code'] = data['code']
                graphs.append(code_graph)
                in_scope_function.add(fidx)
            else:
                if label == 1:
                    vnt += 1
                else:
                    nvnt += 1
        shard_file.close()
        del shard_data
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    output_file = open(os.path.join(output_dir, project + '-' + portion + '.json'), 'w')
    json.dump(graphs, output_file)
    output_file.close()
    print(project, portion, len(total_functions), len(in_scope_function), vnt, nvnt, sep='\t')

def extract_line_graph_data(
    project, base_dir='/home/ding/dlvp/dl-vulnerability-detection/data/commits/code/output/', 
    output_dir='/home/ding/dlvp/dl-vulnerability-detection/data/commits/code/full_experiment_real_data_processed/'):
    wv_model_li = Word2Vec.load('../data/Word2Vec/li_et_al_wv')
    split_dir = f'/home/ding/dlvp/dl-vulnerability-detection/data/commits/code/{project}/raw_code/'
    parsed = f'/home/ding/dlvp/dl-vulnerability-detection/data/commits/code/{project}/parsed/'
    wv_path = '../data/neurips_parsed/raw_code_neurips.100'
    wv_model_original = Word2Vec.load(wv_path)
    shards = os.listdir(os.path.join(base_dir, project))
    shard_count = len(shards)
    total_functions, in_scope_function = set(), set()
    vnt, nvnt = 0, 0
    graphs = []
    for sc in range(1, shard_count + 1):
        shard_file = open(os.path.join(base_dir, project, project + '.json.shard' + str(sc)))
        shard_data = json.load(shard_file)
        try:
            for data in tqdm(shard_data):
                file_name = data['file_name']    
                label = int(file_name.strip()[:-2].split('_')[-1])
                code_text = read_code_file(split_dir + file_name.strip())
                nodes_file_path = parsed + file_name.strip() + '/nodes.csv'
                edges_file_path = parsed + file_name.strip() + '/edges.csv'
                nc = open(nodes_file_path)
                nodes_file = csv.DictReader(nc, delimiter='\t')
                nodes = [node for node in nodes_file]
                if len(nodes) == 0:
                    continue
                nodes = read_csv(nodes_file_path)
                edges = read_csv(edges_file_path)
                node_indices, node_ids, line_numbers, node_id_to_ln = extract_nodes_with_location_info(nodes)
                adjacency_list = create_adjacency_list(line_numbers, node_id_to_ln, edges, False)
                combined_graph = combine_control_and_data_adjacents(adjacency_list)
                data_point = reformat_code_line_graph(code_text, adjacency_list, label, wv_model_original, wv_model_li, label)
                graphs.append(data_point)
        finally:
            pass
        del shard_data
    output_file = open(os.path.join(output_dir, project + '-line_ggnn.json'), 'w')
    json.dump(graphs, output_file)
    output_file.close()
    print(len(graphs))

def main():
    PROJECT = "ReVeal"
    extract_slices(PROJECT)
    print("done function 1\n")
    # create_ggnn_data.create_ggnn_data_main(PROJECT)
    # print("done function 2\n")
    # extract_graph_data("full_graph", PROJECT)
    # print("done function 3\n")
    # extract_line_graph_data(PROJECT)
    # print("done function 4\n")
    # split_data.split_data_main(PROJECT)

if __name__ == "__main__":
    main()
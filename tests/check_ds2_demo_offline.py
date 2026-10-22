"""Offline tests using real routing/validation, mocked WFDB and HTTP transports."""
import ast
import importlib.util
import json
import re
import sys
import tempfile
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
read_paths = []
replies = []
llm_requests = []

class HTTPError(Exception): pass
class Response:
    def __init__(self, data): self.data, self.is_error = data, False
    def json(self): return self.data
    def raise_for_status(self): pass
class Client:
    def __init__(self, **kw): pass
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def post(self, url, **kw):
        if 'openrouter.ai' in url:
            llm_requests.append(kw['json'].copy())
            m = replies.pop(0)
            return Response({'model': 'offline-model', 'choices': [{'message': m, 'finish_reason': 'tool_calls' if m.get('tool_calls') else 'stop'}]})
        data = kw['json']
        selected = int(data['ecg'][0])
        label_index = selected % 5
        scores = {name: float(i == label_index) for i, name in enumerate(['N','S','V','F','Q'])}
        return Response({'mode': 'onnx', 'model_version': '20261008_094922', 'class_order': list(scores), 'processed_count': 2, 'skipped_count': 1, 'beats': [{'r_peak_index': r, 'predicted_class': list(scores)[label_index], 'probabilities': scores} for r in [400,700]], 'skipped': [{'r_peak_index': 20}]})
httpx = types.ModuleType('httpx');httpx.Client=Client;httpx.HTTPError=HTTPError
sys.modules['httpx']=httpx
wfdb=types.ModuleType('wfdb')
wfdb.rdheader=lambda path:SimpleNamespace(sig_name=['MLII'],fs=360,units=['mV'],sig_len=1000)
def rdrecord(path):
    read_paths.append(path)
    return SimpleNamespace(sig_name=['MLII'],p_signal=np.full((1000,1),float(Path(path).name)))
wfdb.rdrecord=rdrecord
wfdb.rdann=lambda path, ext:SimpleNamespace(sample=np.array([20,400,700]),symbol=['N','N','N'])
sys.modules['wfdb']=wfdb
dotenv=types.ModuleType('dotenv');dotenv.load_dotenv=lambda *a,**kw:None;sys.modules['dotenv']=dotenv
spec=importlib.util.spec_from_file_location('agent_tools',ROOT/'agent_tools.py');tools=importlib.util.module_from_spec(spec);sys.modules['agent_tools']=tools;spec.loader.exec_module(tools)
source=ast.parse((ROOT/'web_agent.py').read_text());node=next(n for n in source.body if isinstance(n,ast.FunctionDef) and n.name=='run_conversation')
code=compile(ast.Module(body=[node],type_ignores=[]),'web_agent.py','exec')
class AgentError(RuntimeError): pass
logs=[]
ns={'json':json,'re':re,'httpx':httpx,'AgentError':AgentError,'ToolArgumentError':tools.ToolArgumentError,'normalize_record_id':tools.normalize_record_id,'RECORD_ID':tools.RECORD_ID,'ALLOWED_RECORD_IDS':tools.ALLOWED_RECORD_IDS,'TOOL_SCHEMAS':tools.TOOL_SCHEMAS,'SYSTEM_PROMPT':'test','execute_tool':tools.execute_tool,'audit':lambda *a,**k:logs.append((a,k))}
exec(code,ns)
def classify(r):return {'content':None,'tool_calls':[{'id':'call_1','type':'function','function':{'name':'classify_record','arguments':json.dumps({'record_id':r})}}]}
def stop(s='summary'):return {'content':s}
def run(message,selected,messages):
    replies[:]=messages
    llm_requests.clear()
    return ns['run_conversation'](SimpleNamespace(history=[],message=message,selected_record_id=selected),'test','no-key','offline')

with tempfile.TemporaryDirectory() as folder:
    tools.ROOT=Path(folder)
    (tools.ROOT/'data/mitdb').mkdir(parents=True)
    ds2=json.loads((ROOT/'notebooks/tv2/split.json').read_text())['DS2']['records']
    assert set(tools.ALLOWED_RECORD_IDS)==set(ds2) and len(ds2)==21
    assert set(tools.TOOL_SCHEMAS[1]['function']['parameters']['properties']['record_id']['enum'])==set(ds2)
    print('PASS: 21 allowed IDs and tool schema match historical TV2 split exactly')
    for rid in tools.ALLOWED_RECORD_IDS:
        for ext in ('dat','hea','atr'):(tools.ROOT/'data/mitdb'/f'{rid}.{ext}').touch()
    for rid in tools.ALLOWED_RECORD_IDS:
        result=tools.execute_tool('classify_record',{'record_id':int(rid)})
        assert result['record_id']==rid and result['dataset_split']=='DS2'
        assert Path(read_paths[-1]).name==rid
        assert sum(result['class_counts'].values())==2
    print('PASS: all 21 routes read the requested recording and validate separate API responses')
    for rid in ['101','201','202','102','217','../100','100.hea','00100',True,100.0,None,[]]:
        try:tools.normalize_record_id(rid)
        except tools.ToolArgumentError:pass
        else:raise AssertionError(rid)
    print('PASS: DS1, 202, paced records, paths and malformed IDs rejected')
    for name,args in [('run_shell',{}),('list_records',{'path':'.env'}),('classify_record',{'record_id':'103','path':'.env'})]:
        try:tools.execute_tool(name,args)
        except tools.ToolArgumentError:pass
        else:raise AssertionError((name,args))
    print('PASS: extra fields and unknown tools remain denied')
    (tools.ROOT/'data/mitdb/103.dat').unlink()
    entries=tools.execute_tool('list_records',{})['records']
    assert len(entries)==21 and sum(e['ready'] for e in entries)==20
    assert next(e for e in entries if e['record_id']=='103')['ready'] is False
    try:tools.execute_tool('classify_record',{'record_id':'103'})
    except ValueError:pass
    else:raise AssertionError('missing file accepted')
    print('PASS: one missing recording does not break catalog; classification reports missing data')
    (tools.ROOT/'data/mitdb/103.dat').touch()
    r=run('Phân tích bản đang chọn','103',[classify(103),stop()]);assert r[0]['record_id']=='103'
    print('PASS: selected recording 103 is the target when request has no explicit ID')
    r=run('Phân tích bản ghi 105','103',[classify('103'),classify('105'),stop()]);assert r[0]['record_id']=='105' and any(e['status']=='error' for e in r[3])
    print('PASS: explicit 105 overrides selection 103; wrong model call is rejected')
    before=len(read_paths)
    r=run('Phân loại bản ghi 101','103',[classify('103'),stop('draft'),stop('Không được truy cập 101')]);assert r[0] is None and len(read_paths)==before
    print('PASS: forbidden 101 never substitutes selected 103, no save-eligible API result')
    r=run('Phân loại bản đang chọn','103',[classify('101'),stop('draft'),stop('failed'),classify('103'),stop()]);assert r[0]['record_id']=='103'
    assert any(q['tool_choice']=={'type':'function','function':{'name':'classify_record'}} for q in llm_requests)
    print('PASS: bounded repair uses selected 103 rather than hard-coded 100')
    r=run('Phân loại bản ghi 105',None,[classify('105'),stop()]);assert r[0]['record_id']=='105'
    print('PASS: clients without selector still support explicit authorized ID')
    before=len(read_paths)
    r=run('Phân tích bản ghi 103 và bản ghi 105','103',[classify('103'),stop('draft'),stop('Chọn một bản ghi')]);assert r[0] is None and len(read_paths)==before
    print('PASS: ambiguous multi-record classification does not silently select one')
    r=run('Phân loại bản ghi 202','100',[stop('draft'),stop('Bản 202 không được phép')]);assert r[0] is None and r[3][0]['tool']=='list_records'
    print('PASS: allowlist fallback retained for ignored tool-choice refusal')
print('All offline checks passed. Mocked transports are not real-data ONNX verification.')
# Import the full web module using only a minimal FastAPI routing adapter.
# Pydantic models, orchestration, state, audit and report writes are real.
class Router:
    def __init__(self, **kw): pass
    def get(self, *a, **kw): return lambda fn: fn
    def post(self, *a, **kw): return lambda fn: fn
class HTTPException(Exception):
    def __init__(self, status_code, detail): self.status_code,self.detail=status_code,detail
fastapi=types.ModuleType('fastapi');fastapi.APIRouter=Router;fastapi.HTTPException=HTTPException;fastapi.Request=type('Request',(),{})
sys.modules['fastapi']=fastapi
spec=importlib.util.spec_from_file_location('web_agent',ROOT/'web_agent.py');web=importlib.util.module_from_spec(spec);spec.loader.exec_module(web)
with tempfile.TemporaryDirectory() as folder:
    tools.ROOT=web.ROOT=Path(folder)
    (tools.ROOT/'data/mitdb').mkdir(parents=True)
    for rid in tools.ALLOWED_RECORD_IDS:
        for ext in ('dat','hea','atr'):(tools.ROOT/'data/mitdb'/f'{rid}.{ext}').touch()
    web.os.environ['OPENROUTER_API_KEY']='offline-placeholder'
    request=SimpleNamespace(headers={'origin':'http://127.0.0.1:8000','host':'127.0.0.1:8000'})
    config=web.config();assert config['ready_count']==21 and len(config['records'])==21
    replies[:]=[classify('103'),stop('Kết quả bản ghi 103')]
    data=web.chat(web.ChatInput(message='Phân tích bản đang chọn',selected_record_id='103'),request)
    assert data['api_result']['record_id']=='103' and data['save_token']
    assert not (web.ROOT/'outputs').exists()
    decision=web.ReportDecision(run_id=data['run_id'],token=data['save_token'],action='save',confirmed=False)
    try:web.report_decision(decision,request)
    except HTTPException as error:assert error.status_code==422
    else:raise AssertionError('unconfirmed save accepted')
    decision=web.ReportDecision(run_id=data['run_id'],token=data['save_token'],action='save',confirmed=True)
    saved=web.report_decision(decision,request)
    report=json.loads((web.ROOT/'outputs'/saved['filename']).read_text())
    assert report['confirmed'] is True and report['api_result']['record_id']=='103' and report['selected_record_id']=='103'
    print('PASS: selector config and real report-confirmation handlers preserve selected/actual record and require confirmation')
    before=set((web.ROOT/'outputs').iterdir())
    replies[:]=[classify('105'),stop('Kết quả bản ghi 105')]
    data=web.chat(web.ChatInput(message='Phân loại bản ghi 105',selected_record_id='103'),request)
    assert data['api_result']['record_id']=='105'
    discarded=web.report_decision(web.ReportDecision(run_id=data['run_id'],token=data['save_token'],action='discard'),request)
    assert discarded['saved'] is False and set((web.ROOT/'outputs').iterdir())==before
    print('PASS: discard creates no report file; explicit record overrides selector without mixing results')
    try:web.chat(web.ChatInput(message='Phân tích',selected_record_id='101'),request)
    except HTTPException as error:assert error.status_code==422
    else:raise AssertionError('unauthorized selector accepted')
    print('PASS: forged selector outside DS2 rejected before LLM or data access')
print('All 15 test groups passed with stubbed network/WFDB. Real ONNX checks remain to be run on laptop.')

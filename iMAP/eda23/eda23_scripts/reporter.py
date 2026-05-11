
import re
import os

class Reporter(object):
    def __init__(self, cases, workspace) -> None:
        self.cases = cases
        self.workspace = workspace

    def report(self):
        result= [] 
        for case in self.cases:
            case_name = os.path.split(case)[1]
            rltfile = os.path.join(self.workspace, case_name, case_name+'.rlt')
            if not os.path.isfile(rltfile):
                result.append(f'{case_name},0,no_rlt_error')
                continue
            with open(rltfile) as f:
                content = f.read()
            result.append(content)
        
        # sort by aig size
        result.sort(key=lambda x: int(x.split(',')[1]))
        result.insert(0, ','.join(['case', 'size', 'depth', 'timeout', 'gen_time', 'check_time', 'area', 'level']))
        rpt_str = '\n'.join(result)
        rpt_file = os.path.join(self.workspace, 'report.csv')
        with open(rpt_file, 'w') as f:
            f.write(rpt_str)
        
        return rpt_str
        
        
        





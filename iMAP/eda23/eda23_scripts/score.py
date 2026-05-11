
class Case(object):
    def __init__(self, result_str):
        self.name = ''
        self.size = 0
        self.depth = 0
        self.timeout = 0
        self.gen_time = 0
        self.check_time = 0
        self.area = 0
        self.level = 0

        results = result_str.strip().split(',')
        self.name = results[0]
        if 'error' in result_str:
            self.area = 'error'
            self.level = 'error'
        elif 'timeout' in result_str:
            self.area = 'timeout'
            self.level = 'timeout'
        else:
            self.size, self.depth, self.timeout, self.gen_time, self.check_time, self.area, self.level = results[1:]

    def is_timeout(self):
        return self.area == 'timeout'
    
    def is_error(self):
        return self.area == 'error'
    
    def set_area_score(self, score):
        self.area_score = score

    def set_level_score(self, score):
        self.level_score = score

    def calculate_score(self):
        self.score = self.level_score * 0.6 + self.area_score * 0.4
        self.weight_score = self.score
        if 'middle' in self.name or 'attach' in self.name:
            self.weight_score = self.score * 2
        elif 'large' in self.name:
            self.weight_score = self.score * 3



class Team(object):
    def __init__(self, team, rlt_file):
        self.name = team
        cases = []
        with open(rlt_file) as f:
            for line in f:
                if 'case,size,depth,timeout' in line: continue # header
                if 'case54' in line or 'case57' in line or 'case59' in line: continue
                cases.append(Case(line))
        
        # construct dict
        self.cases = {}
        for case in cases:
            self.cases[case.name] = case

    def get_case(self, case_name):
        return self.cases[case_name]

    def ratio(self):
        ratio = {}
        ratio['eda230209'] = 1.04
        ratio['eda230211'] = 1.05
        ratio['eda230234'] = 1.03
        ratio['eda230239'] = 1.05
        ratio['eda230256'] = 1.03
        if self.name in ratio:
            return ratio[self.name]
        return 1.0

    def calculate_score(self):
        self.score = sum([case.weight_score for case in self.cases.values()])
        self.weight_score = self.score * self.ratio()

class Score(object):
    def __init__(self, team_list_file) -> None:
        self.teams = []
        with open(team_list_file) as f:
            for line in f:
                team = line.strip()
                if '#' in team:
                    print(f'skip team {team}')
                    continue
                rlt_file = f'/home/{team}/checking_results/report.csv'
                self.teams.append(Team(team, rlt_file))

    def calculate_score(self):
        team_number = len(self.teams)
        # for each case
        for case_name in self.teams[0].cases.keys():
            # collect all teams
            case_dict = {}
            area_list = []
            level_list = []
            for team in self.teams:
                case = team.get_case(case_name)
                case_dict[team.name] = case
                # area score
                if case.is_timeout() or case.is_error():
                    case.set_area_score(team_number)
                    case.set_level_score(team_number)
                    case.calculate_score()
                else:
                    area_list.append(case.area)
                    level_list.append(case.level)
            
            area_list.sort(key=int)
            level_list.sort(key=int)
            #if 'large_case3' in case_name:
            #    print(area_list)
            #    print(level_list)
            for case in case_dict.values():
                if case.is_timeout() or case.is_error(): continue
                case.set_area_score(area_list.index(case.area) + 1)
                case.set_level_score(level_list.index(case.level) + 1)
                case.calculate_score()
                #if 'large_case3' in case_name:
                #    print(f'area: {case.area}, area_score: {case.area_score}')
                #    print(f'level: {case.level}, level_score: {case.level_score}')

    def report(self):
        # header
        header1 = ['ID']
        header2 = ['item']
        for team in self.teams:
            header1 += [team.name] * 6
            header2 += ['level', 'area', 'level_score', 'area_score', 'score', 'weight_score']
        
        # case
        case_results = []
        # for each case
        #for case_name in self.teams[0].cases.keys().sorted():
        case_names = sorted(list(self.teams[0].cases.keys()))
        for case_name in case_names:
            result = [case_name]
            for team in self.teams:
                case = team.get_case(case_name)
                result.extend([case.level, case.area, str(case.level_score), str(case.area_score), f'{case.score:.1f}', f'{case.weight_score:.1f}'])
            case_results.append(','.join(result))
        
        case_results.insert(0, ','.join(header2))
        case_results.insert(0, ','.join(header1))


        # report total score
        score_list = []
        result = ['score']
        for team in self.teams:
            team.calculate_score()
            score_list.append(team.weight_score)
            result.extend(['', '', '', '', '', f'{team.score:.1f}'])
        case_results.append(','.join(result))

        result = ['ratio']
        for team in self.teams:
            result.extend(['', '', '', '', '', f'{team.ratio():.2f}'])
        case_results.append(','.join(result))

        result = ['final_score']
        for team in self.teams:
            result.extend(['', '', '', '', '', f'{team.weight_score:.3f}'])
        case_results.append(','.join(result))

        score_list.sort()
        for team in self.teams:
            print(f'{team.name}: {team.weight_score:.3f}, {score_list.index(team.weight_score)+1}')

        csv_str = '\n'.join(case_results)
        with open('score.csv', 'w') as f:
            f.write(csv_str)


if __name__ == '__main__':
    import sys
    s = Score(sys.argv[1])
    s.calculate_score()
    s.report()

        

                
                




            
        



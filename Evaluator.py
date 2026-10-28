import torch
import TicTacToe
import TicTacToeAgent

def run_game(model1, model2, shouldPrint=False):
    ttt = TicTacToe.TicTacToe()
    info = {}
    
    while ttt.Winner == 0:
        result = False
        info["mask"] = ttt.get_mask()
        
        if ttt.turn == 1:
            info["Model1Move"] = TicTacToeAgent.selectAction(model1, ttt.convert_board(), info["mask"])
            attempted_move = info["Model1Move"]
            result = ttt.play(attempted_move)
        else:
            info["Model2Move"] = TicTacToeAgent.selectAction(model2, ttt.convert_board(), info["mask"])
            attempted_move = info["Model2Move"]
            result = ttt.play(attempted_move)
        
        info["board"] = ttt.board
        info["turn"] = ttt.turn
        
        if shouldPrint: print(ttt.board)
            
        assert result is True, (
            f"\n--- Invalid Move Error ---\n"
            f"-> Turn: Player {info['turn']}\n"
            f"-> Attempted Move Index: {attempted_move}\n"
            f"-> Valid Move Mask (1=Valid, 0=Invalid): {info['mask']}\n"
            f"-> Current Board State: {info['board']}\n"
            f"-> Full Info Dictionary: {info}"
        )
    
    return ttt.Winner

def multi_game(model1, model2, game_count, shouldPrint=False, PrintInterval=25):
    evaluation = {"model1" : {"wins" : 0, "draws" : 0, "losses" : 0}, 
                  "model2" : {"wins" : 0, "draws" : 0, "losses" : 0}}
    
    players = {1 : "model1", 2 : "model2"}
    
    for match in range(game_count):
        result = run_game(model1, model2, shouldPrint and match % PrintInterval == 0)
        if result != -1:
            winner = players[result]
            loser = "model2" if winner == "model1" else "model1"
            
            evaluation[winner]["wins"] += 1
            evaluation[loser]["losses"] += 1
        else:
            evaluation["model1"]["draws"] += 1
            evaluation["model2"]["draws"] += 1
        
        model1, model2 = model2, model1
        players[1], players[2] = players[2], players[1]
        
    return evaluation

def evaluate_champion(champion, device, opp_count=3, seed=1000, game_count=100, shouldPrint=False, PrintInterval=25):
    evaluation = [{"wins" : 0, "draws" : 0, "losses" : 0, "fitness" : 0} for _ in range(opp_count)]
    
    for i in range(opp_count):
        torch.manual_seed(seed+i)
        opponent = TicTacToeAgent.TicTacToeAI().to(device)
        multieval = multi_game(champion, opponent, game_count,
                                shouldPrint=shouldPrint, PrintInterval=PrintInterval)
        evaluation[i]["wins"] += multieval["model1"]["wins"]
        evaluation[i]["draws"] += multieval["model1"]["draws"]
        evaluation[i]["losses"] += multieval["model1"]["losses"]
    
    for i in range(opp_count):
        torch.manual_seed(seed+i)
        opponent = TicTacToeAgent.TicTacToeAI().to(device)
        multieval = multi_game(opponent, champion, game_count,
                                shouldPrint=shouldPrint, PrintInterval=PrintInterval)
        evaluation[i]["wins"] += multieval["model2"]["wins"]
        evaluation[i]["draws"] += multieval["model2"]["draws"]
        evaluation[i]["losses"] += multieval["model2"]["losses"]
        evaluation[i]["fitness"] = (evaluation[i]["wins"] + 0.5 * evaluation[i]["draws"]) / (game_count * 2)
    
    return evaluation

def evaluate_champions(champs, device, opp_count=3, game_count=100,
                       shouldPrint=False, PrintInterval=25, seed=1000,
                       include_baseline=False):
    """
    Evaluate a list of champions (+ optional baseline).
    
    Returns
    -------
    evaluations : list of evaluation lists
    names       : list of display names (same order as evaluations)
    """
    evaluations = []
    names = []

    # Champions
    for i, champ in enumerate(champs):
        evals = evaluate_champion(champ, device, opp_count, seed, game_count,
                                  shouldPrint, PrintInterval)
        evaluations.append(evals)
        names.append(f"Champ {i+1}")

    # Optional baseline (fresh frozen model)
    if include_baseline:
        print("Evaluating baseline (fresh frozen model)...")
        baseline = TicTacToeAgent.TicTacToeAI().to(device)
        baseline.eval()
        evals = evaluate_champion(baseline, device, opp_count, seed, game_count,
                                  shouldPrint, PrintInterval)
        evaluations.append(evals)
        names.append("Baseline")

    return evaluations, names
    
        
    
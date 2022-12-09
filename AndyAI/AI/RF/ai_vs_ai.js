const config = {
  draggable: true,
  position: 'start',
  onDragStart: onDragStart,
  onDrop: onDrop,
  onSnapEnd: onSnapEnd
}

const game = new Chess();
const board = Chessboard('myBoard', config);

const pawnEvalWhite =
  [
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
    [1.0, 1.0, 2.0, 3.0, 3.0, 2.0, 1.0, 1.0],
    [0.5, 0.5, 1.0, 2.5, 2.5, 1.0, 0.5, 0.5],
    [0.0, 0.0, 0.0, 2.0, 2.0, 0.0, 0.0, 0.0],
    [0.5, -0.5, -1.0, 0.0, 0.0, -1.0, -0.5, 0.5],
    [0.5, 1.0, 1.0, -2.0, -2.0, 1.0, 1.0, 0.5],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
  ];

const pawnEvalBlack = reverseArray(pawnEvalWhite);

const knightEval =
  [
    [-5.0, -4.0, -3.0, -3.0, -3.0, -3.0, -4.0, -5.0],
    [-4.0, -2.0, 0.0, 0.0, 0.0, 0.0, -2.0, -4.0],
    [-3.0, 0.0, 1.0, 1.5, 1.5, 1.0, 0.0, -3.0],
    [-3.0, 0.5, 1.5, 2.0, 2.0, 1.5, 0.5, -3.0],
    [-3.0, 0.0, 1.5, 2.0, 2.0, 1.5, 0.0, -3.0],
    [-3.0, 0.5, 1.0, 1.5, 1.5, 1.0, 0.5, -3.0],
    [-4.0, -2.0, 0.0, 0.5, 0.5, 0.0, -2.0, -4.0],
    [-5.0, -4.0, -3.0, -3.0, -3.0, -3.0, -4.0, -5.0]
  ];

const bishopEvalWhite =
  [
    [-2.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -2.0],
    [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0],
    [-1.0, 0.0, 0.5, 1.0, 1.0, 0.5, 0.0, -1.0],
    [-1.0, 0.5, 0.5, 1.0, 1.0, 0.5, 0.5, -1.0],
    [-1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, -1.0],
    [-1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0],
    [-1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.5, -1.0],
    [-2.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -2.0]
  ];

const bishopEvalBlack = reverseArray(bishopEvalWhite);

const rookEvalWhite =
  [
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5],
    [-0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.5],
    [-0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.5],
    [-0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.5],
    [-0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.5],
    [-0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.5],
    [0.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.0]
  ];

const rookEvalBlack = reverseArray(rookEvalWhite);

const queenEval =
  [
    [-2.0, -1.0, -1.0, -0.5, -0.5, -1.0, -1.0, -2.0],
    [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0],
    [-1.0, 0.0, 0.5, 0.5, 0.5, 0.5, 0.0, -1.0],
    [-0.5, 0.0, 0.5, 0.5, 0.5, 0.5, 0.0, -0.5],
    [0.0, 0.0, 0.5, 0.5, 0.5, 0.5, 0.0, -0.5],
    [-1.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.0, -1.0],
    [-1.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, -1.0],
    [-2.0, -1.0, -1.0, -0.5, -0.5, -1.0, -1.0, -2.0]
  ];

const kingEvalWhite =
  [
    [-3.0, -4.0, -4.0, -5.0, -5.0, -4.0, -4.0, -3.0],
    [-3.0, -4.0, -4.0, -5.0, -5.0, -4.0, -4.0, -3.0],
    [-3.0, -4.0, -4.0, -5.0, -5.0, -4.0, -4.0, -3.0],
    [-3.0, -4.0, -4.0, -5.0, -5.0, -4.0, -4.0, -3.0],
    [-2.0, -3.0, -3.0, -4.0, -4.0, -3.0, -3.0, -2.0],
    [-1.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -1.0],
    [2.0, 2.0, 0.0, 0.0, 0.0, 0.0, 2.0, 2.0],
    [2.0, 3.0, 1.0, 0.0, 0.0, 1.0, 3.0, 2.0]
  ];

const kingEvalBlack = reverseArray(kingEvalWhite);

function reverseArray(array) {
  return array.slice().reverse();
}

function onDragStart(source, piece, position, orientation) {
  if (game.game_over()) return false;

  if (piece.search(/^b/) !== -1) {
    return false;
  }
}

function onDrop(source, target) {
  const move = game.move({
    from: source,
    to: target,
    promotion: 'q'
  });

  if (move === null) return 'snapback';

  let timeslot = 2000
  // 50 steps decide win loss
  for (let j = 0; j < 50; j++) {
    setTimeout(makeBestMove, timeslot);
    timeslot += 6000
    setTimeout(makeBestMove2, timeslot);
    timeslot += 5000
  }
}

function onSnapEnd() {
  board.position(game.fen());
}

function makeBestMove() {
  if (game.game_over()) {
    alert('Game over');
  }
  const bestMove = minimaxRoot(game, 2, true);
  game.move(bestMove);
  board.position(game.fen());
  if (game.game_over()) {
    alert('Game over');
  }
}
function minimax(game, depth, alpha, beta, maximizingPlayer) {
  if (depth === 0) {
    return -evaluateBoard(game.board());
  }

  // returns all feasible moves in that position
  const moves = game.moves();

  if (maximizingPlayer) {
    let bestMove = -Infinity;
    for (const move of moves) {
      game.move(move);
      const value = minimax(game, depth - 1, alpha, beta, false);
      bestMove = Math.max(bestMove, value);
      alpha = Math.max(alpha, value);
      if (alpha >= beta) {
        game.undo();
        break;
      }
      game.undo();
    }
    return bestMove;
  } else {
    let bestMove = +Infinity;
    for (const move of moves) {
      game.move(move);
      const value = minimax(game, depth - 1, alpha, beta, true);
      bestMove = Math.min(bestMove, value);
      beta = Math.min(beta, value);
      if (alpha >= beta) {
        game.undo();
        break;
      }
      game.undo();
    }
    return bestMove;
  }
}
function minimaxRoot(game, depth, maximizingPlayer) {
  const moves = game.moves();
  let bestMove = -Infinity;
  let bestMoveFound = null;

  for (const move of moves) {
    game.move(move);
    const value = minimax(game, depth - 1, -Infinity, Infinity, !maximizingPlayer);
    game.undo();
    if (value >= bestMove) {
      bestMove = value;
      bestMoveFound = move;
    }
  }

  return bestMoveFound;
}
function evaluateBoard(board) {
  let totalEvaluation = 0;
  for (let i = 0; i < 8; i++) {
    for (let j = 0; j < 8; j++) {
      totalEvaluation += getPieceValue(board[i][j], i, j);
    }
  }
  return totalEvaluation;
}
function getPieceValue(piece, x, y) {
  if (piece === null) {
    return 0;
  }
  let absoluteValue;
  if (piece.type === 'p') {
    absoluteValue = 30 + (piece.color ? pawnEvalWhite[x][y] : pawnEvalBlack[x][y]);
  } else if (piece.type === 'n') {
    absoluteValue = 30 + knightEval[x][y];
  } else if (piece.type === 'b') {
    absoluteValue = 30 + (piece.color ? bishopEvalWhite[x][y] : bishopEvalBlack[x][y]);
  } else if (piece.type === 'r') {
    absoluteValue = 30 + (piece.color ? rookEvalWhite[x][y] : rookEvalBlack[x][y]);
  } else if (piece.type === 'q') {
    absoluteValue = 100 + queenEval[x][y];
  } else if (piece.type === 'k') {
    absoluteValue = 900 + (piece.color ? kingEvalWhite[x][y] : kingEvalBlack[x][y]);
  } else {
    throw Error(`Unknown piece type: ${piece.type}`);
  }

  return piece.color === 'w' ? absoluteValue : -absoluteValue;
}

// for me
// ---------------------------------------------------------------
function makeBestMove2() {
  if (game.game_over()) {
    alert('Game over');
  }
  const bestMove = minimaxRoot2(game, 3, true);
  game.move(bestMove);
  board.position(game.fen());
  if (game.game_over()) {
    alert('Game over');
  }
}
function minimax2(game, depth, alpha, beta, maximizingPlayer) {
  if (depth === 0) {
    return -evaluateBoard2(game.board());
  }

  // returns all feasible moves in that position
  const moves = game.moves();

  if (maximizingPlayer) {
    let bestMove = -Infinity;
    for (const move of moves) {
      game.move(move);
      const value = minimax2(game, depth - 1, alpha, beta, false);
      bestMove = Math.max(bestMove, value);
      alpha = Math.max(alpha, value);
      if (alpha >= beta) {
        game.undo();
        break;
      }
      game.undo();
    }
    return bestMove;
  } else {
    let bestMove = +Infinity;
    for (const move of moves) {
      game.move(move);
      const value = minimax2(game, depth - 1, alpha, beta, true);
      bestMove = Math.min(bestMove, value);
      beta = Math.min(beta, value);
      if (alpha >= beta) {
        game.undo();
        break;
      }
      game.undo();
    }
    return bestMove;
  }
}
function minimaxRoot2(game, depth, maximizingPlayer) {
  const moves = game.moves();
  let bestMove = -Infinity;
  let bestMoveFound = null;

  for (const move of moves) {
    game.move(move);
    const value = minimax2(game, depth - 1, -Infinity, Infinity, !maximizingPlayer);
    game.undo();
    if (value >= bestMove) {
      bestMove = value;
      bestMoveFound = move;
    }
  }

  return bestMoveFound;
}
function evaluateBoard2(board) {
  let totalEvaluation = 0;
  for (let i = 0; i < 8; i++) {
    for (let j = 0; j < 8; j++) {
      totalEvaluation += getPieceValue2(board[i][j], i, j);
    }
  }
  return totalEvaluation;
}
function getPieceValue2(piece, x, y) {
  if (piece === null) {
    return 0;
  }
  let absoluteValue;
  if (piece.type === 'p') {
    absoluteValue = 10 + (piece.color ? pawnEvalWhite[x][y] : pawnEvalBlack[x][y]);
  } else if (piece.type === 'n') {
    absoluteValue = 40 + knightEval[x][y];
  } else if (piece.type === 'b') {
    absoluteValue = 50 + (piece.color ? bishopEvalWhite[x][y] : bishopEvalBlack[x][y]);
  } else if (piece.type === 'r') {
    absoluteValue = 50 + (piece.color ? rookEvalWhite[x][y] : rookEvalBlack[x][y]);
  } else if (piece.type === 'q') {
    absoluteValue = 90 + queenEval[x][y];
  } else if (piece.type === 'k') {
    absoluteValue = 900 + (piece.color ? kingEvalWhite[x][y] : kingEvalBlack[x][y]);
  } else {
    throw Error(`Unknown piece type: ${piece.type}`);
  }

  return piece.color === 'b' ? absoluteValue : -absoluteValue;
}